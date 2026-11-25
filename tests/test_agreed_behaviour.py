"""What was agreed about this package, each as a test that names it.

The engine is Soniox and nothing here can call Whisper; the transcript is what was said and a
second reader never replaces a word; nobody is named from how much they speak; a folder named in
VOICEMEMOS_EXCLUDE_FOLDERS (patient conversations) is outside everything; and what is done on top
of the engine's own transcription is timed and held against a tenth of it.
"""
import json
import re
from pathlib import Path

from tests.conftest import FakeTranscriber
from voicememos_mcp import align, notation
from voicememos_mcp.config import DEFAULTS, load_config
from voicememos_mcp.worker import Worker

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "voicememos_mcp"


def test_the_engine_is_soniox_v6_and_nothing_can_call_whisper():
    assert DEFAULTS["SONIOX_MODEL"] == "stt-async-v6"
    assert 'SONIOX_MODEL="stt-async-v6"' in (ROOT / "config.example.env").read_text().replace("'", '"') \
        or "stt-async-v6" in (ROOT / "config.example.env").read_text()
    for path in sorted(SRC.glob("*.py")):
        assert not re.search(r"whisper", path.read_text(), re.I), f"{path.name} names Whisper"
    assert not re.search(r"whisper", (ROOT / "pyproject.toml").read_text(), re.I)
    # every transcription of the worker goes through the one Soniox client
    assert re.findall(r"^from \.soniox import .*$|^\s+from \.soniox import .*$", (SRC / "worker.py").read_text(), re.M)


def _primary(texts, confs=None):
    confs = confs or [0.99] * len(texts)
    return [{"text": t, "conf": c, "start_ms": i * 500, "end_ms": i * 500 + 400, "lang": "fi"}
            for i, (t, c) in enumerate(zip(texts, confs))]


def test_the_transcript_is_what_was_said_and_a_second_reader_never_replaces_a_word():
    said = ["mä", "niinku", "meen", "sinne", "huomenna."]
    # the reader "tidies" the speech into what was meant: standard language, a filler dropped, a word added
    words, _ = align.align(_primary(said, [0.99, 0.6, 0.5, 0.99, 0.99]), "minä menen sinne varmaan huomenna.")
    assert [w["text"] for w in words if w.get("text")] == said  # the engine's words, in order, all of them
    text = notation.render_blocks(words, 0.7)[0]
    for word in ("mä", "niinku", "meen", "sinne", "huomenna"):
        assert word in text
    # what the reader heard is shown beside the engine's word, after it, never in its place
    for group in re.findall(r"\{([^}]*)\}", text):
        first = group.split("|")[0]
        assert first == notation.EMPTY or all(part.strip(notation.PUNCTUATION) in [s.strip(notation.PUNCTUATION) for s in said]
                                              for part in first.split())


def test_nobody_is_named_from_how_much_they_speak(cfg, library):
    # Two speakers, one with nearly all the words, the owner's name configured, no voiceprint enrolled:
    # the one who speaks most stays S1.
    tokens = [{"text": f" w{i}", "start_ms": i * 1000, "end_ms": i * 1000 + 900, "confidence": 0.99, "speaker": "1"}
              for i in range(40)] + [{"text": " joo", "start_ms": 41000, "end_ms": 41500, "confidence": 0.99, "speaker": "2"}]
    labels = notation.speaker_labels(notation.tokens_to_words(tokens)[0])
    assert labels["1"]["share"] > 0.9 and {v["label"] for v in labels.values()} == {"S1", "S2"}
    env = dict(cfg.raw, VOICEMEMOS_VOICEPRINT_NAME="Alex")
    named = load_config(path=cfg.data_dir / "missing.env", env=env)
    Worker(named, transcriber=FakeTranscriber(tokens=tokens)).run(only=["aaaa1111"])
    md = next(named.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "**S1:**" in md and "**S2:**" in md and "Alex" not in md
    data = json.loads(next(named.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["voiceprint"] is None and all("basis" not in v for v in data["speakers"].values())


def test_a_patient_folder_named_in_exclude_folders_is_never_sent_written_or_listed(cfg, fake):
    # conftest: the memo "Secret plans" is in the folder "Private", and VOICEMEMOS_EXCLUDE_FOLDERS names it.
    worker = Worker(cfg, transcriber=fake)
    worker.run()
    assert not any("EEEE5555" in name for name in fake.calls)  # never sent to the engine
    assert not list(cfg.output_dir.rglob("*eeee5555*"))  # no transcript, no sidecar
    assert not (cfg.data_dir / "tmp" / "EEEE5555-0000-0000-0000-000000000005").exists()
    for name in ("library.json", "worker.json", "state.json"):  # not even its title in what other programs read
        assert "Secret plans" not in (cfg.data_dir / name).read_text()
    memo = next(m for m in worker._memos() if m.id.startswith("EEEE5555"))
    assert worker.transcribe_memo(memo, force=True) == {"id": memo.id, "short_id": memo.short_id, "title": None,
                                                        "recorded": memo.recorded, "status": "excluded"}
    assert not any("EEEE5555" in name for name in fake.calls)


class Ticks:
    """A clock for the stage timer: each call moves on by the next step."""

    def __init__(self, steps):
        self.steps, self.now = list(steps), 0.0

    def __call__(self):
        self.now += self.steps.pop(0) if self.steps else 0.0
        return self.now


def _reader(text):
    def read(audio):
        return {"text": text, "model": "gemini-4-argon", "usage": {}, "finish_reason": "STOP"}
    return read


def test_the_extra_stages_are_timed_against_the_engines_own_time(cfg, fake):
    env = dict(cfg.raw, VOICEMEMOS_SECOND_READER="gemini")
    with_reader = load_config(path=cfg.data_dir / "missing.env", env=env)
    worker = Worker(with_reader, transcriber=fake, second_reader=_reader("Hello world. Idea: build a bird house."))
    # timer calls: before the engine, after it (100 s), before the reader, after it (5 s), before the voiceprint
    worker.timer = Ticks([0, 100, 0, 5, 0])
    worker.run(only=["aaaa1111"])
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["timing"] == {"primary_s": 100.0, "extras_s": {"second_reader": 5.0}, "extras_total_s": 5.0,
                              "extras_share": 0.05, "limit": 0.1, "over": False}
    assert "extras_slow" not in {a["kind"] for a in json.loads((cfg.data_dir / "worker.json").read_text())["alerts"]}
    usage = next(iter(json.loads((cfg.data_dir / "usage.json").read_text())["months"].values()))
    assert usage["primary_s"] == 100.0 and usage["extras_s"] == 5.0

    slow = Worker(with_reader, transcriber=fake, second_reader=_reader("Hello world. Idea: build a bird house."))
    slow.timer = Ticks([0, 100, 0, 30, 0])
    slow.run(only=["bbbb2222"])
    data = json.loads(next(cfg.output_dir.rglob("*car-notes_bbbb2222.json")).read_text())
    assert data["timing"]["extras_share"] == 0.3 and data["timing"]["over"] is True
    alert = next(a for a in json.loads((cfg.data_dir / "worker.json").read_text())["alerts"] if a["kind"] == "extras_slow")
    assert "30 %" in alert["line"] and "10 %" in alert["line"] and "Car notes" not in alert["line"]


def test_the_share_is_absent_and_never_zero_when_the_engines_time_is_not_known(cfg, fake):
    worker = Worker(cfg, transcriber=fake)
    assert worker._timing(None, {}) is None
    assert worker._timing(None, {"voiceprint": 2.0}) == {"extras_s": {"voiceprint": 2.0}, "extras_total_s": 2.0}
    assert "extras_share" not in worker._timing(0.0, {"voiceprint": 2.0})
