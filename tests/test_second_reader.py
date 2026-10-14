import json

from tests.conftest import FakeTranscriber
from voicememos_mcp import align, notation
from voicememos_mcp.gemini import Gemini, build_prompt
from voicememos_mcp.report import build_report, render_report
from voicememos_mcp.store import Store
from voicememos_mcp.tools import Tools
from voicememos_mcp.worker import Worker


def primary(texts, confs=None):
    confs = confs or [0.99] * len(texts)
    return [{"text": t, "conf": c, "start_ms": i * 500, "end_ms": i * 500 + 400, "lang": "en"}
            for i, (t, c) in enumerate(zip(texts, confs))]


def render(words):
    return notation.render_blocks(words, 0.7)[0]


def test_identical_readers_add_no_marks():
    words, m = align.align(primary(["Hello", "there."]), "hello there")
    assert render(words) == "Hello there." and m["agreement"] == 1.0 and m["disagreement_spans"] == 0


def test_single_word_substitution():
    words, m = align.align(primary(["build", "a", "bird", "house."], [0.99, 0.99, 0.5, 0.99]),
                           "build a bed house.")
    assert render(words) == "build a {bird|bed} house."
    assert words[2]["basis"] == "reader_disagreement"
    assert m["substituted"] == 1 and m["spans_with_low_primary_confidence"] == 1


def test_span_deletion_and_insertion():
    words, m = align.align(primary(["one", "two", "three", "four", "five."]), "one three four five six.")
    assert render(words) == "one {two|∅} three four five. {∅|six}"
    assert (m["deleted"], m["inserted"]) == (1, 1)
    words, _ = align.align(primary(["a", "b", "c", "d."]), "a x y d.")
    assert render(words) == "a {b c|x y} d."
    words, m = align.align(primary(["go", "now."]), "go home now.")
    assert render(words) == "go {∅|home} now."
    assert m["inserted"] == 1


def test_reader_marks_are_parsed():
    words = align.reader_words("I think (?) so [?] yes")
    assert words[1] == {"text": "think", "unsure": True}
    assert words[3] == {"text": "", "gap": True}


def test_dominant_language():
    assert align.dominant_language([{"lang": "fi"}] * 9 + [{"lang": "en"}]) == "fi"
    assert align.dominant_language([{"lang": "fi"}] * 6 + [{"lang": "en"}] * 4) == "mixed"


def test_prompt_mentions_languages_and_terms():
    prompt = build_prompt(["fi", "en"], ["Roam"], "Notes recorded in a car.")
    assert "Finnish, English" in prompt and "Roam" in prompt and "Never translate" in prompt


class FakeResponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._data, self.headers = status, data or {}, headers or {}
        self.ok = status < 400
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0)

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0)

    def delete(self, url, **kw):
        self.calls.append(("DELETE", url, kw))
        return FakeResponse()


def test_gemini_inline_request(tmp_path, monkeypatch):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"\0" * 10)
    answer = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "Hello there."}]},
                              "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 42},
              "modelVersion": "gemini-4-argon"}
    session = FakeSession([FakeResponse(503), FakeResponse(200, answer)])
    monkeypatch.setattr("voicememos_mcp.gemini.time.sleep", lambda s: None)
    result = Gemini("k", session=session).transcribe(audio, "prompt")
    assert result["text"] == "Hello there." and result["usage"]["totalTokenCount"] == 42
    body = session.calls[-1][2]["json"]
    assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/mp4"
    assert session.headers["x-goog-api-key"] == "k"


def test_gemini_large_file_is_uploaded_and_deleted(tmp_path, monkeypatch):
    import voicememos_mcp.gemini as g

    monkeypatch.setattr(g, "INLINE_LIMIT", 5)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"\0" * 10)
    session = FakeSession([
        FakeResponse(200, {}, {"x-goog-upload-url": "https://upload.example/u1"}),
        FakeResponse(200, {"file": {"name": "files/abc", "uri": "https://files/abc", "state": "ACTIVE",
                                    "mimeType": "audio/mp4"}}),
        FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}),
    ])
    assert g.Gemini("k", session=session).transcribe(audio, "p")["text"] == "ok"
    assert session.calls[-1][0] == "DELETE" and session.calls[-1][1].endswith("/v1beta/files/abc")


def fake_reader(text="Hello world. Idea: build a bed house."):
    calls = []

    def read(audio):
        calls.append(audio.name)
        return {"text": text, "model": "gemini-4-argon", "usage": {"totalTokenCount": 100}, "finish_reason": "STOP"}

    read.calls = calls
    return read


def test_second_reader_inline_with_first_transcription(cfg, fake):
    reader = fake_reader()
    Worker(cfg, transcriber=fake, second_reader=reader).run()
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "{bird|bed}" in md and "Second reader: gemini-4-argon" in md
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert [r["role"] for r in data["readers"]] == ["primary", "second"] and data["language"] == "en"
    library = json.loads((cfg.data_dir / "library.json").read_text())
    walk = next(m for m in library["memos"] if m["title"] == "Morning walk")
    assert walk["transcript"]["second_reader"]["model"] == "gemini-4-argon"


def test_second_read_later_without_primary_call(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    calls_before = len(fake.calls)
    reader = fake_reader()
    summary = Worker(cfg, transcriber=fake, second_reader=reader).run_second_read(only=["aaaa1111"])
    assert summary["items"][0]["status"] == "second_read"
    assert len(fake.calls) == calls_before and len(reader.calls) == 1
    again = Worker(cfg, transcriber=fake, second_reader=reader).run_second_read(only=["aaaa1111"])
    assert again["items"][0]["status"] == "already_read"
    redo = Worker(cfg, transcriber=fake, second_reader=fake_reader("Hello world. Idea: build a bird house."))
    assert redo.run_second_read(only=["aaaa1111"], redo=True)["items"][0]["agreement"]["agreement"] == 1.0
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "{" not in md.split("## Transcript")[1]


def test_second_reader_failure_keeps_primary(cfg, fake):
    def broken(audio):
        raise RuntimeError("quota")

    Worker(cfg, transcriber=fake, second_reader=broken).run()
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["readers"][1]["error"] == "quota" and "Hello world (?)." in render_markdown_text(cfg)


def render_markdown_text(cfg):
    return next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()


def test_queue_second_read_and_report(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    tools = Tools(cfg)
    job = tools.second_read(since="2000-01-01")
    Worker(cfg, transcriber=fake, second_reader=fake_reader()).run()
    status = tools.transcribe_status(job["job_id"])
    assert status["status"] == "done" and [i["status"] for i in status["items"]] == ["second_read", "second_read"]
    report = build_report(Store(cfg.data_dir, cfg.output_dir))
    en = report["languages"]["en"]
    assert en["memos"] == 2 and en["disagreements_per_100_words"] > 0
    assert en["samples"][0]["primary"] == "bird" and en["samples"][0]["reader"] == "bed"
    text = render_report(report)
    assert "| English |" in text and "A (primary): bird" in text
    result = tools.compare_readers()
    assert result["report_file"].endswith(".md") and "samples" not in result["languages"]["en"]


def test_queued_second_read_respects_limit_and_skips_done(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    tools = Tools(cfg)
    first = tools.second_read(since="2000-01-01", limit=1)
    reader = fake_reader()
    Worker(cfg, transcriber=fake, second_reader=reader).run()
    assert tools.transcribe_status(first["job_id"])["total"] == 1
    second = tools.second_read(since="2000-01-01")
    Worker(cfg, transcriber=fake, second_reader=reader).run()
    assert tools.transcribe_status(second["job_id"])["total"] == 1  # the already-read memo is skipped
    assert len(reader.calls) == 2
