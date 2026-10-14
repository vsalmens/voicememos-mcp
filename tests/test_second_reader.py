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
    order = [(x["recorded"], x["memo_id"], x["start_ms"]) for x in en["samples"]]
    assert order == sorted(order)
    assert sum(1 for line in text.splitlines() if line.startswith("### ")) == len({x["memo_id"] for x in en["samples"]})
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


def fi_words(texts, confs=None):
    return [{**w, "lang": "fi"} for w in primary(texts, confs)]


def test_finnish_spoken_forms_are_not_disagreements():
    words, m = align.align(fi_words(["Mä", "oon", "sit", "tulossa", "niinku", "huomenna."]),
                           "Minä olen sitten tulossa niinkuin huomenna.")
    assert m["disagreement_spans"] == 0 and m["agreement"] == 1.0 and m["spoken_forms_folded"] == 4
    assert render(words) == "Mä oon sit tulossa niinku huomenna."


def test_finnish_folding_keeps_real_differences():
    _, m = align.align(fi_words(["mä", "huomasin", "sen."]), "minä huomasi sen.")
    assert m["disagreement_spans"] == 1 and m["substituted"] == 1


def test_english_sun_is_not_folded():
    _, m = align.align(primary(["the", "sun", "is", "up."]), "the sinun is up.")
    assert m["disagreement_spans"] == 1


def test_gemini_low_reads_only_uncertain_memos(cfg, fake):
    cfg.raw["VOICEMEMOS_SECOND_READER"] = "gemini-low"
    reader = fake_reader()
    worker = Worker(cfg, transcriber=fake, second_reader=reader)
    sure = primary(["all", "clear", "here."])
    unsure = primary(["mumble", "mumble", "maybe."], [0.3, 0.4, 0.99])
    assert not worker.wants_second_read(sure) and worker.wants_second_read(unsure)
    cfg.raw["VOICEMEMOS_SECOND_READER"] = "gemini"
    assert worker.wants_second_read(sure)


def long_words(minutes=70, every_s=10):
    """One 3-word sentence every ``every_s`` seconds, each word 0.5 s, sentence ends with a period."""
    out = []
    for n in range(minutes * 60 // every_s):
        base = n * every_s * 1000
        for k, t in enumerate(["alpha", "beta", f"gamma{n}."]):
            out.append({"text": t, "conf": 0.99, "start_ms": base + k * 600, "end_ms": base + k * 600 + 500,
                        "lang": "en"})
    return out


def test_plan_chunks_cuts_between_sentences_into_equal_parts():
    words = long_words(70)
    chunks = align.plan_chunks(words, 30 * 60_000)
    assert len(chunks) == 2 and chunks[0]["first"] == 0 and chunks[-1]["last"] == len(words)
    cut = chunks[1]["first"]
    assert words[cut - 1]["text"].endswith(".") and words[cut]["text"] == "alpha"
    assert words[cut - 1]["end_ms"] < chunks[0]["end_ms"] < words[cut]["start_ms"]
    assert abs(chunks[0]["end_ms"] - 35 * 60_000) < 5 * 60_000 and chunks[1]["end_ms"] is None
    assert len(align.plan_chunks(words[:30], 30 * 60_000)) == 1


def test_long_memo_read_in_parts_only_uncertain(cfg, tmp_path):
    from types import SimpleNamespace

    words = long_words(70)
    first_cut = align.plan_chunks(words, 30 * 60_000)[1]["first"]
    for w in words[first_cut:]:
        w["conf"] = 0.3  # the second half is uncertain
    cuts, reads = [], []

    def cutter(audio, start_s, end_s, out):
        cuts.append((start_s, end_s))
        out.write_text("clip")
        return out

    def reader(clip):
        reads.append(clip.name)
        return {"text": " ".join(w["text"] for w in words[first_cut:]).replace("beta", "bravo", 1),
                "model": "gemini-4-argon", "usage": {"totalTokenCount": 7}}

    worker = Worker(cfg, transcriber=None, second_reader=reader, audio_cutter=cutter)
    audio = tmp_path / "long.m4a"
    audio.write_text("x")
    memo = SimpleNamespace(id="long", duration_s=70 * 60)
    out, record = worker._apply_second_reader(reader, audio, memo, words, only_uncertain=True)
    assert len(reads) == 1 and cuts[0][1] is None and cuts[0][0] > 30 * 60
    assert [c["text"] is None for c in record["chunks"]] == [True, False]
    assert record["agreement"]["chunks_read"] == 1 and record["agreement"]["substituted"] == 1
    assert len(out) == len(words) and out[first_cut + 1]["alts"] == ["bravo"]
    out_all, record_all = worker._apply_second_reader(reader, audio, memo, words, only_uncertain=False)
    assert len(reads) == 3 and record_all["agreement"]["chunks_read"] == 2


def test_on_demand_only_uncertain_skips_clear_memos(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    reader = fake_reader()
    summary = Worker(cfg, transcriber=fake, second_reader=reader).run_second_read(since="2000-01-01",
                                                                                  only_uncertain=True)
    library = json.loads((cfg.data_dir / "library.json").read_text())
    certainty = {m["short_id"]: (m.get("transcript") or {}).get("stats", {}).get("certainty")
                 for m in library["memos"]}
    for item in summary["items"]:
        if certainty.get(item["short_id"]) == "low":
            assert item["status"] == "second_read"
        else:
            assert item["status"] == "skipped" and item["error"].startswith("certainty")
    assert len(reader.calls) == sum(1 for i in summary["items"] if i["status"] == "second_read")
    assert summary["items"]


def test_reading_that_does_not_match_is_rejected_and_old_marks_removed(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    good = fake_reader()
    Worker(cfg, transcriber=fake, second_reader=good).run_second_read(only=["aaaa1111"])
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "{bird|bed}" in md
    made_up = fake_reader("Warm up for ten minutes. Squats: three sets of twelve. Plank thirty seconds.")
    summary = Worker(cfg, transcriber=fake, second_reader=made_up).run_second_read(only=["aaaa1111"], redo=True)
    assert summary["items"][0]["status"] == "rejected"
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "{" not in md.split("## Transcript", 1)[1]
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    second = [r for r in data["readers"] if r["role"] == "second"][0]
    assert second["error"].startswith("rejected") and "Squats" in second["text"]


def test_truncated_reading_leaves_the_rest_uncompared():
    words = primary(["one", "two", "three", "four", "five", "six."])
    out, m = align.align(words, "one two three", truncated=True)
    assert render(out) == "one two three four five six."
    assert m["uncovered_words"] == 3 and m["deleted"] == 0 and m["agreement"] == 1.0
    _, full = align.align(words, "one two three")
    assert full["deleted"] == 3


def test_one_refused_part_does_not_lose_the_others(cfg, tmp_path):
    from types import SimpleNamespace

    from voicememos_mcp.gemini import GeminiError

    words = long_words(70)
    cut = align.plan_chunks(words, 30 * 60_000)[1]["first"]
    calls = []

    def cutter(audio, start_s, end_s, out):
        out.write_text("clip")
        return out

    def reader(clip):
        calls.append(clip.name)
        if clip.name.startswith("part1"):
            raise GeminiError("Gemini returned no candidates: {'blockReason': 'OTHER'}")
        return {"text": " ".join(w["text"] for w in words[cut:]), "model": "gemini-4-argon",
                "usage": {"totalTokenCount": 1}, "finish_reason": "STOP"}

    worker = Worker(cfg, transcriber=None, second_reader=reader, audio_cutter=cutter)
    audio = tmp_path / "long.m4a"
    audio.write_text("x")
    out, record = worker._apply_second_reader(reader, audio, SimpleNamespace(id="l", duration_s=4200), words)
    assert "error" not in record and len(calls) == 2
    assert record["chunks"][0]["failed"].startswith("Gemini") and record["agreement"]["chunks_read"] == 1
