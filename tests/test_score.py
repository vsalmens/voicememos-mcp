import json

from tests.test_second_reader import FakeResponse, FakeSession, fake_reader
from voicememos_mcp import score as sc
from voicememos_mcp.gemini import GeminiTranscribe, interaction_text
from voicememos_mcp.worker import Worker

FLAT = """## Samples: English (gemini-4-argon)

- [ ] **2026-04-03T20:01** Saunalahti 2 `ce5d4a78` at 4:59, primary confidence 0.68
  - A (primary): things,
  - B (reader): ∅
  - context: get clarity on more specific ▮ like time, start and end
  - better:  B
- [x] **2026-04-03T20:01** Saunalahti 2 `ce5d4a78` at 1:01, primary confidence 0.56
  - A (primary): Mila
  - B (reader): Milla
  - context: and I hope you and ▮ and your crowd are doing
  - better: A (from pronunciation, from context: B)
- [ ] **2018-07-13T02:51** 1 `dc681362` at 0:01, primary confidence 0.25
  - A (primary): Tere heppa,
  - B (reader): Dear Heba,
  - context:  ▮ good evening. I hope you
  - better: A / B / neither
"""

GROUPED = """## Samples: Finnish (gemini-4-argon)


### Saunalahti, 2020-06-24 01:56 `05fb1a00`

- [ ] at 4:12, primary confidence 0.97
  - A (primary): insight.
  - B (reader): insightti.
  - context: toisen olla turvassa. Aika jännä ▮ 
  - better: B (both)
- [ ] at 8:47, primary confidence 0.98
  - A (primary): nään
  - B (reader): näen
  - context: minuuttia siihen. Toivottavasti, että mä ▮ sut ja toivottavasti saan taas
  - better: A
- [ ] at 2:35
  - A (primary): ∅
  - B (reader): Yhtäkkiä
  - context: vaihtoi ilmettä sitten niinku varoituksena. ▮ Siellä on joku varoitus, että
  - better: neither
"""


def test_parse_both_layouts_and_skip_unmarked():
    spots = sc.parse_labels(FLAT) + sc.parse_labels(GROUPED)
    assert [(s.memo, s.at, sorted(s.accept)) for s in spots] == [
        ("ce5d4a78", "4:59", ["B"]), ("ce5d4a78", "1:01", ["A"]),
        ("05fb1a00", "4:12", ["A", "B"]), ("05fb1a00", "8:47", ["A"])]
    assert spots[0].lang == "en" and spots[2].lang == "fi" and spots[0].reading("B") == []


def test_judge_finds_the_spot_by_context():
    spot = sc.parse_labels(FLAT)[0]
    right = sc.tokens("I want to get clarity on more specific, like time, start and end of it.")
    wrong = sc.tokens("I want to get clarity on more specific things like time, start and end.")
    assert sc.judge(spot, right).correct is True and sc.judge(spot, right).matches == "B"
    verdict = sc.judge(spot, wrong)
    assert verdict.correct is False and verdict.said == ["things"]
    assert sc.judge(spot, sc.tokens("Something else entirely here.")).correct is None


def test_spoken_forms_are_not_folded():
    spot = sc.parse_labels(GROUPED)[1]
    assert sc.judge(spot, sc.tokens("viittäkään minuuttia siihen. Toivottavasti, että mä näen sut ja")).correct is False
    assert sc.judge(spot, sc.tokens("minuuttia siihen. Toivottavasti että mä nään sut ja toivottavasti")).correct is True


def test_score_every_reading_of_a_transcript():
    spot = sc.parse_labels(FLAT)[0]
    data = {"engine": {"model": "stt-async-v6"},
            "words": [{"text": t} for t in "to get clarity on more specific things like time, start and end".split()],
            "readers": [{"role": "primary"},
                        {"role": "second", "model": "gemini-4-argon", "text": "get clarity on more specific like time, start and end"},
                        {"role": "candidate", "requested_model": "gemini-4-neon", "text": "get clarity on more specific stuff like time, start and end"}]}
    scores = sc.score([spot], {"ce5d4a78": data})
    assert {n: s.counts()["correct"] for n, s in scores.items()} == {
        "soniox (stt-async-v6)": 0, "gemini-4-argon (active)": 1, "gemini-4-neon": 0}
    assert "| gemini-4-argon (active) | 1/1 | 1/1 | 0 |" in sc.render(scores)


def test_candidate_reading_leaves_the_transcript_alone(cfg, fake):
    Worker(cfg, transcriber=fake).run()
    Worker(cfg, transcriber=fake, second_reader=fake_reader()).run_second_read(only=["aaaa1111"])
    md_path = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md"))
    json_path = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json"))
    before_md, before_words = md_path.read_text(), json.loads(json_path.read_text())["words"]
    worker = Worker(cfg, transcriber=fake)
    worker.reader_factory = lambda cfg_, model: fake_reader("Hello world. Idea: build a bard house.")
    summary = worker.run_second_read(only=["aaaa1111"], model="gemini-4-neon", candidate=True)
    assert summary["items"][0]["status"] == "candidate_read"
    data = json.loads(json_path.read_text())
    assert data["words"] == before_words and md_path.read_text().split("## Transcript")[1] == before_md.split("## Transcript")[1]
    roles = [(r["role"], r.get("requested_model")) for r in data["readers"]]
    assert ("second", None) in roles and ("candidate", "gemini-4-neon") in roles
    again = worker.run_second_read(only=["aaaa1111"], model="gemini-4-neon", candidate=True)
    assert again["items"][0]["status"] == "already_read"
    redo = worker.run_second_read(only=["aaaa1111"], model="gemini-4-neon", candidate=True, redo=True)
    assert redo["items"][0]["status"] == "candidate_read"
    assert sum(r["role"] == "candidate" for r in json.loads(json_path.read_text())["readers"]) == 1


def test_interaction_text_shapes():
    sample = {"id": "interactions/1", "status": "completed",
              "steps": [{"type": "model_output", "content": [
                  {"type": "text", "text": "Hei, mitä kuuluu?", "annotations": []}]}]}
    assert interaction_text(sample) == "Hei, mitä kuuluu?"
    assert interaction_text({"output_text": " Moi "}) == "Moi"


def test_transcribe_request_and_language_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("voicememos_mcp.gemini.time.sleep", lambda s: None)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"\0" * 10)
    done = {"id": "interactions/9", "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Mä oon tässä."}]}]}
    session = FakeSession([
        FakeResponse(200, {}, {"x-goog-upload-url": "https://upload.example/u"}),
        FakeResponse(200, {"file": {"name": "files/f", "uri": "https://files/f", "state": "ACTIVE",
                                    "mimeType": "audio/mp4"}}),
        FakeResponse(400, {"error": {"message": "only one language code is supported"}}),
        FakeResponse(200, done),
    ])
    reader = GeminiTranscribe("k", languages=["fi", "en"], vocabulary=["Liminaali"], session=session)
    result = reader.transcribe(audio)
    assert result["text"] == "Mä oon tässä."
    posts = [c for c in session.calls if c[0] == "POST" and c[1].endswith("/v1beta/interactions")]
    first, second = posts[0][2]["json"], posts[1][2]["json"]
    config = first["generation_config"]["transcription_config"]
    assert first["model"] == "gemini-3.5-transcribe" and first["input"][0]["uri"] == "https://files/f"
    assert config["language_codes"] == ["fi-FI", "en-US"] and config["mode"] == {"type": "verbatim"}
    assert config["custom_vocabulary"] == ["Liminaali"]
    assert "language_codes" not in second["generation_config"]["transcription_config"]
    assert session.calls[-1][0] == "DELETE"
