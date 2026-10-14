import json

from tests.conftest import FakeTranscriber
from voicememos_mcp import notation
from voicememos_mcp.soniox import Soniox
from voicememos_mcp.worker import Worker

CONVERSATION = [
    {"text": "How", "start_ms": 0, "end_ms": 200, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": " are", "start_ms": 200, "end_ms": 300, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": " you", "start_ms": 300, "end_ms": 400, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": "?", "start_ms": 400, "end_ms": 410, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": " Fine", "start_ms": 600, "end_ms": 800, "confidence": 0.5, "speaker": "2", "language": "en"},
    {"text": ".", "start_ms": 800, "end_ms": 810, "confidence": 0.99, "speaker": "2", "language": "en"},
    {"text": " Good", "start_ms": 1000, "end_ms": 1200, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": " to", "start_ms": 1200, "end_ms": 1300, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": " hear", "start_ms": 1300, "end_ms": 1500, "confidence": 0.99, "speaker": "1", "language": "en"},
    {"text": ".", "start_ms": 1500, "end_ms": 1510, "confidence": 0.99, "speaker": "1", "language": "en"},
]


def words():
    return notation.tokens_to_words(CONVERSATION)[0]


def test_single_speaker_gets_no_labels():
    solo = [dict(w, speaker="1") for w in words()]
    assert notation.speaker_labels(solo) == {}
    assert notation.speaker_labels([dict(w, speaker=None) for w in words()]) == {}


def test_speakers_numbered_and_rendered_per_turn():
    labels = notation.speaker_labels(words())
    assert {k: v["label"] for k, v in labels.items()} == {"1": "S1", "2": "S2"}
    assert notation.render_blocks(words(), 0.7, speakers=labels) == [
        "**S1:** How are you?", "**S2:** Fine (?).", "**S1:** Good to hear."]


def test_owner_guess_is_marked_uncertain():
    labels = notation.speaker_labels(words(), owner_name="Alex", owner_min_share=0.6)
    assert labels["1"]["label"] == "Alex (?)" and labels["1"]["basis"] == "speaking_share"
    assert labels["2"]["label"] == "S2"
    # no dominant speaker -> no owner guess
    even = notation.speaker_labels(words()[:4], owner_name="Alex", owner_min_share=0.8)
    assert all("Alex" not in v["label"] for v in even.values())


def test_worker_stores_speakers_and_renders_them(cfg):
    cfg.raw["VOICEMEMOS_OWNER_NAME"] = "Alex"
    Worker(cfg, transcriber=FakeTranscriber(tokens=CONVERSATION)).run()
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "**Alex (?):** How are you?" in md and "**S2:** Fine (?)." in md
    assert "guess from speaking share" in md
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["speakers"]["2"]["share"] > 0 and data["words"][0]["speaker"] == "1"
    library = json.loads((cfg.data_dir / "library.json").read_text())
    assert next(m for m in library["memos"] if m["title"] == "Morning walk")["transcript"]["speakers"] == 2


class Recorder:
    def __init__(self):
        self.headers, self.bodies = {}, []

    def request(self, method, url, **kw):
        if "json" in kw:
            self.bodies.append(kw["json"])

        class R:
            ok, status_code, text = True, 200, ""

            def __init__(self, data):
                self._d = data

            def json(self):
                return self._d

        if url.endswith("/v1/files"):
            return R({"id": "f1"})
        if url.endswith("/v1/transcriptions"):
            return R({"id": "t1"})
        if url.endswith("/transcript"):
            return R({"tokens": []})
        return R({"status": "completed"})

    def delete(self, url, **kw):
        class R:
            ok, status_code = True, 200
        return R()


def test_soniox_request_enables_diarization(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"0")
    rec = Recorder()
    Soniox("k", session=rec, poll_interval=0).transcribe(audio, ["fi", "en"], diarization=True)
    assert rec.bodies[0]["enable_speaker_diarization"] is True
    rec2 = Recorder()
    Soniox("k", session=rec2, poll_interval=0).transcribe(audio, ["fi"])
    assert "enable_speaker_diarization" not in rec2.bodies[0]
