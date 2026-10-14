"""Voiceprint logic with synthetic audio and a fake embedder (no model needed).

Each synthetic "voice" is a tone at its own frequency; the fake embedder returns the
spectrum energy in a few bands, so voices separate the way real embeddings do.
"""
import json

import pytest

np = pytest.importorskip("numpy")

from tests.conftest import FakeTranscriber  # noqa: E402
from voicememos_mcp import notation, voiceprint  # noqa: E402
from voicememos_mcp.tools import ToolError, Tools  # noqa: E402
from voicememos_mcp.worker import Worker  # noqa: E402

SR = voiceprint.SAMPLE_RATE
BANDS = [(100, 220), (220, 340), (340, 520), (520, 800)]


def tone(freq, seconds, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def fake_embed(samples):
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / SR)
    return np.array([spectrum[(freqs >= a) & (freqs < b)].sum() for a, b in BANDS] + [1.0], dtype=np.float32)


OWNER, OTHER = 180, 420


def words_for(spans):
    """spans: list of (speaker, start_s, end_s) -> one word per second."""
    out = []
    for speaker, a, b in spans:
        for sec in range(int(a), int(b)):
            out.append({"text": "w", "start_ms": sec * 1000, "end_ms": sec * 1000 + 900, "conf": 0.99,
                        "speaker": speaker})
    return out


def test_enroll_and_identify_speakers():
    memos = {f"m{i}": tone(OWNER + i * 3, 20, seed=i, noise=0.05 + 0.02 * i) for i in range(4)}
    vp = voiceprint.enroll("Alex", "test-model", memos, fake_embed)
    assert vp["thresholds"]["match"] > vp["thresholds"]["maybe"]
    assert len(vp["own_scores"]) == 4 and not vp["warnings"]

    conversation = np.concatenate([tone(OWNER, 8, seed=10), tone(OTHER, 8, seed=11)])
    words = words_for([("1", 0, 8), ("2", 8, 16)])
    speakers = notation.speaker_labels(words)
    updated, summary = voiceprint.identify(conversation, words, speakers, vp, fake_embed)
    assert updated["1"]["label"] == "Alex" and updated["1"]["basis"] == "voiceprint"
    assert updated["2"]["label"] == "S2" and updated["2"]["voiceprint"] < updated["1"]["voiceprint"]
    assert summary["result"] == "match" and summary["speaker"] == "1"

    _, solo = voiceprint.identify(tone(OWNER, 10, seed=20), words_for([("1", 0, 10)]), {}, vp, fake_embed)
    assert solo["result"] == "match"
    _, other = voiceprint.identify(tone(OTHER, 10, seed=21), [], {}, vp, fake_embed)
    assert other["result"] == "no match"


def test_two_other_people_are_not_named():
    memos = {f"m{i}": tone(OWNER, 15, seed=i) for i in range(3)}
    vp = voiceprint.enroll("Alex", "test-model", memos, fake_embed)
    audio = np.concatenate([tone(OTHER, 8, seed=1), tone(650, 8, seed=2)])
    words = words_for([("1", 0, 8), ("2", 8, 16)])
    updated, summary = voiceprint.identify(audio, words, notation.speaker_labels(words), vp, fake_embed)
    assert summary["result"] == "no match"
    assert {v["label"] for v in updated.values()} == {"S1", "S2"}


def test_enrollment_needs_enough_material():
    with pytest.raises(voiceprint.VoiceprintError):
        voiceprint.enroll("Alex", "m", {"a": tone(OWNER, 10, 1)}, fake_embed)
    with pytest.raises(voiceprint.VoiceprintError):
        voiceprint.enroll("Alex", "m", {"a": tone(OWNER, 10, 1), "b": tone(OWNER, 1.5, 2)}, fake_embed)


def test_voiceprint_file_is_private(tmp_path):
    store = voiceprint.Voiceprint(tmp_path / "voiceprint.json")
    store.save({"name": "Alex", "centroid": [0.1], "enrolled": [{"memo_id": "x", "windows": 3}]})
    assert oct((tmp_path / "voiceprint.json").stat().st_mode & 0o777) == "0o600"
    summary = store.summary()
    assert summary["enrolled"] and "centroid" not in summary


def test_worker_enrolls_and_identifies_through_the_queue(cfg, library):
    conv_tokens = [
        {"text": " Hello", "start_ms": 0, "end_ms": 900, "confidence": 0.99, "speaker": "1"},
        {"text": " there", "start_ms": 1000, "end_ms": 1900, "confidence": 0.99, "speaker": "1"},
        {"text": " friend", "start_ms": 2000, "end_ms": 7900, "confidence": 0.99, "speaker": "1"},
        {"text": " Hi", "start_ms": 8000, "end_ms": 8900, "confidence": 0.99, "speaker": "2"},
        {"text": " back", "start_ms": 9000, "end_ms": 15900, "confidence": 0.99, "speaker": "2"},
    ]
    owner_audio = {"AAAA1111": tone(OWNER, 20, seed=1), "BBBB2222": tone(OWNER + 4, 20, seed=2, noise=0.08)}

    def lazy(path, audio):
        """Like the real reader: decodes from the file on demand, so the file must still exist."""
        def read(start_s, duration_s):
            assert path.exists(), f"{path.name} was removed before it was read"
            a = int(max(start_s, 0) * SR)
            return audio[a: a + int(duration_s * SR)]
        return read

    def loader(path):
        for key, audio in owner_audio.items():
            if key in path.name:
                return lazy(path, audio)
        return lazy(path, np.concatenate([tone(OWNER, 8, seed=5), tone(OTHER, 8, seed=6)]))

    worker = lambda tokens=None: Worker(cfg, transcriber=FakeTranscriber(tokens=tokens),  # noqa: E731
                                        embedder=fake_embed, audio_loader=loader)
    worker().run()  # transcribe (no voiceprint yet)
    tools = Tools(cfg)
    assert tools.voiceprint("status")["enrolled"] is False
    with pytest.raises(ToolError):
        tools.voiceprint("identify", since="2000-01-01")
    with pytest.raises(ToolError):
        tools.voiceprint("enroll", memos=["aaaa1111"], name="Alex")

    job = tools.voiceprint("enroll", memos=["aaaa1111", "bbbb2222"], name="Alex")
    worker().run()
    done = tools.transcribe_status(job["job_id"])
    assert done["status"] == "done" and done["voiceprint"]["name"] == "Alex"
    assert "centroid" not in json.dumps(done)
    status = tools.voiceprint("status")
    assert status["enrolled"] and status["thresholds"]["match"] > 0

    # a new conversation memo is identified as it is transcribed
    Worker(cfg, transcriber=FakeTranscriber(tokens=conv_tokens), embedder=fake_embed,
           audio_loader=lambda p: np.concatenate([tone(OWNER, 8, seed=5), tone(OTHER, 8, seed=6)])
           ).run(only=["aaaa1111"], redo=True)
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "**Alex:** Hello there friend" in md and "**S2:** Hi back" in md
    assert "voiceprint match" in md

    # identify on demand for an existing transcript
    job = tools.voiceprint("identify", memos=["bbbb2222"])
    worker().run()
    item = tools.transcribe_status(job["job_id"])["items"][0]
    assert item["status"] == "identified" and item["voiceprint"]["result"] == "match"
    md = next(cfg.output_dir.rglob("*car-notes_bbbb2222.md")).read_text()
    assert "Voiceprint: matches Alex" in md


def test_suggestions_prefer_diarized_single_speaker_memos_across_years():
    memos = [
        {"short_id": "a", "recorded": "2019-01-01T10:00:00", "duration_s": 120, "local": True,
         "transcript": {"json": "x", "speakers": 0}},
        {"short_id": "b", "recorded": "2024-01-01T10:00:00", "duration_s": 300, "local": True,
         "transcript": {"json": "y", "speakers": 0}},
        {"short_id": "c", "recorded": "2024-02-01T10:00:00", "duration_s": 300, "local": True,
         "transcript": {"json": "z", "speakers": 2}},
        {"short_id": "d", "recorded": "2024-03-01T10:00:00", "duration_s": 10, "local": True,
         "transcript": {"json": "w", "speakers": 0}},
        {"short_id": "e", "recorded": "2020-03-01T10:00:00", "duration_s": 100, "local": True,
         "transcript": {"json": "v"}},
    ]
    assert [m["short_id"] for m in voiceprint.suggest_enrollment(memos)] == ["a", "b"]
