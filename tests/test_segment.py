"""Memos longer than the engine accepts: planning, merging, the worker flow in parts."""
import json
import logging
import sqlite3
import subprocess

import pytest

from tests.conftest import FakeTranscriber
from voicememos_mcp import notation, segment
from voicememos_mcp.config import load_config
from voicememos_mcp.store import render_markdown
from voicememos_mcp.worker import Worker

HAS_FFMPEG = bool(segment.find_tool("ffmpeg") and segment.find_tool("ffprobe"))
LONG_ID = "AAAA1111-0000-0000-0000-000000000001"  # "Morning walk" in the test library


# ----------------------------------------------------------------------------- planning

def no_silence(audio, a, b):
    return []


def test_fewest_parts_within_the_limit():
    # The memo that started this: 8 h 20 min 40 s with parts of at most 298 - 5 min.
    bounds = segment.plan_bounds(30040.8, 293 * 60, 0, silences=no_silence)
    assert bounds == [0.0, 15020.4, 30040.8]  # N = 2, about 250 min each
    assert segment.part_count(293 * 60, 293 * 60) == 1
    assert segment.part_count(293 * 60 + 1, 293 * 60) == 2
    assert len(segment.plan_bounds(1000, 300, 0, silences=no_silence)) == 5  # 4 parts of 250 s


def test_cut_moves_to_the_longest_silence_in_the_window():
    seen = []

    def silences(audio, a, b):
        seen.append((a, b))
        return [(14000, 14010), (15100, 15104), (15500, 15530)]

    bounds = segment.plan_bounds(30040.8, 293 * 60, 600, silences=silences)
    assert seen == [(14420.4, 15620.4)]
    assert bounds[1] == 15515.0  # middle of the 30 s silence; the one at 14000 is outside the window


def test_cut_never_makes_a_part_too_long():
    # 100 s in parts of at most 60 s: the cut must stay within 40..60 even if a silence is nearby.
    bounds = segment.plan_bounds(100, 60, 20, silences=lambda audio, a, b: [(62, 68)])
    assert bounds == [0.0, 50.0, 100.0]
    assert all(b - a <= 60 for a, b in zip(bounds, bounds[1:]))


# ----------------------------------------------------------------------------- merging

def toks(*items):
    """(text, start_ms, end_ms, confidence, speaker) -> Soniox-style tokens."""
    return [{"text": t, "start_ms": a, "end_ms": b, "confidence": c, "speaker": s, "language": "fi"}
            for t, a, b, c, s in items]


PART1 = toks(("Hei", 0, 300, 0.99, "1"), (" vaan", 350, 600, 0.40, "1"), (".", 600, 610, 0.9, "1"),
             (" Moi", 2000, 2300, 0.95, "2"), (".", 2300, 2310, 0.9, "2"))
PART2 = toks(("Jatketaan", 0, 500, 0.98, "1"), (" tästä", 600, 900, 0.30, "1"), (".", 900, 910, 0.9, "1"))


def test_merge_shifts_times_and_keeps_parts_apart():
    words, had_conf, tokens, ranges = segment.merge_parts([{"start_s": 0, "tokens": PART1},
                                                           {"start_s": 3600.5, "tokens": PART2}])
    assert had_conf
    assert [w["text"] for w in words] == ["Hei", "vaan.", "Moi.", "Jatketaan", "tästä."]
    assert words[3]["start_ms"] == 3_600_500 and words[4]["end_ms"] == 3_601_410
    assert [w["speaker"] for w in words] == ["1:1", "1:1", "1:2", "2:1", "2:1"]
    # raw tokens: same shift, and each part's range is recorded
    assert ranges == [[0, 5], [5, 8]]
    assert tokens[5]["text"] == "Jatketaan" and tokens[5]["start_ms"] == 3_600_500
    assert PART2[0]["start_ms"] == 0  # the inputs are not modified


def test_merged_stats_are_the_sum_of_the_parts():
    w1, _ = notation.tokens_to_words(PART1)
    w2, _ = notation.tokens_to_words(PART2)
    merged, *_ = segment.merge_parts([{"start_s": 0, "tokens": PART1}, {"start_s": 100, "tokens": PART2}])
    s1, s2, s = (notation.stats(x, 0.7) for x in (w1, w2, merged))
    assert s["words"] == s1["words"] + s2["words"] == 5
    assert s["uncertain"] == s1["uncertain"] + s2["uncertain"] == 2
    assert s["marked_share"] == round((s1["uncertain"] + s2["uncertain"]) / 5, 3)  # weighted by words


def test_speaker_labels_are_per_part():
    words, *_ = segment.merge_parts([{"start_s": 0, "tokens": PART1}, {"start_s": 100, "tokens": PART2}])
    labels = segment.speaker_labels(words)
    assert {k: v["label"] for k, v in labels.items()} == {"1:1": "S1a", "1:2": "S2a", "2:1": "S1b"}
    assert sum(v["words"] for v in labels.values()) == 5
    # One speaker in every part: no labels, like a single-speaker memo.
    solo = [{"text": "a", "speaker": "1:1"}, {"text": "b", "speaker": "2:1"}]
    assert segment.speaker_labels(solo) == {}
    assert segment.part_letter(0) == "a" and segment.part_letter(26) == "aa"


def test_split_words_by_bounds():
    words = [{"start_ms": 0}, {"start_ms": 99_999}, {"start_ms": 100_000}, {"start_ms": None}]
    parts = segment.split_words(words, [0, 100, 200])
    assert [len(p) for p in parts] == [3, 1]


def test_markdown_of_a_memo_in_parts():
    words, had_conf, *_ = segment.merge_parts([{"start_s": 0, "tokens": PART1},
                                               {"start_s": 3700, "tokens": PART2}])
    data = {"memo": {"id": "X", "title": "Long", "recorded": "2026-10-24T13:49:51+03:00", "duration_s": 7400},
            "engine": {"model": "stt-async-v5"}, "confidence_available": had_conf, "words": words,
            "speakers": segment.speaker_labels(words),
            "segments": {"count": 2, "bounds_s": [0, 3700.0, 7400.0], "source_token_ranges": [[0, 5], [5, 8]]},
            "readers": [{"role": "primary"}, {"role": "second", "skipped": "too long", "error": "skipped: too long"}]}
    md = render_markdown(data, 0.7, stamp_every_s=60)
    front = md.split("---")[1]
    assert 'engine: "stt-async-v5"' in front
    assert "segments: 2" in front and "segment_bounds_s: [0, 3700.0, 7400.0]" in front
    assert "Transcribed in 2 parts (cut at 1:01:40)" in md
    assert "S2a and S2b are not necessarily the same person" in md
    assert "Second reader skipped: too long." in md
    assert "## Transcript\n\n**S1a:** Hei vaan (?)." in md  # the first paragraph has no stamp, as always
    assert "[1:01:40] **S1b:** Jatketaan tästä (?)." in md  # time stamps continue across the cut


def test_ordinary_markdown_is_unchanged_by_the_new_fields():
    words, _ = notation.tokens_to_words(PART2)
    md = render_markdown({"memo": {"id": "X", "title": "Short", "duration_s": 2}, "engine": {"model": "m"},
                          "confidence_available": True, "words": words}, 0.7)
    assert "segments" not in md and "parts" not in md


# ----------------------------------------------------------------------------- worker

def make_long_memo(library, tmp_path, duration=23.0):
    """Turn "Morning walk" into a real 23 s recording: tone, 3 s silence, tone."""
    path = library / "20261012 080000-AAAA1111.m4a"
    path.unlink()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=10",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=3", "-f", "lavfi", "-i", "sine=f=660:d=10",
                    "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", "-c:a", "aac", str(path)], check=True)
    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZDURATION=? WHERE ZUNIQUEID=?", (duration, LONG_ID))
    db.commit()
    db.close()
    old = path.stat().st_mtime - 3600
    import os

    os.utime(path, (old, old))
    return path


def long_cfg(tmp_path, library, **extra):
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_EXCLUDE_FOLDERS": "private",
           "VOICEMEMOS_WORKER": "none",
           # parts of at most 18 - 3 = 15 s: the 23 s memo needs 2, cut in the silence at 10-13 s
           "VOICEMEMOS_MAX_DURATION_MIN": "0.3", "VOICEMEMOS_SEGMENT_MARGIN_MIN": "0.05",
           "VOICEMEMOS_SEGMENT_SILENCE_WINDOW_MIN": "0.1", **extra}
    return load_config(path=tmp_path / "missing.env", env=env)


def library_status(cfg, memo_id=LONG_ID):
    library = json.loads((cfg.data_dir / "library.json").read_text())
    return next(m for m in library["memos"] if m["id"] == memo_id)


needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are needed")


@needs_ffmpeg
def test_long_memo_is_transcribed_in_two_parts(tmp_path, library):
    make_long_memo(library, tmp_path)
    cfg = long_cfg(tmp_path, library)
    fake = FakeTranscriber()
    summary = Worker(cfg, transcriber=fake).run()

    assert sorted(fake.calls) == ["20261014 090000-BBBB2222.qta", "part1.m4a", "part2.m4a"]
    item = next(i for i in summary["transcribed"] if i["short_id"] == "aaaa1111")
    assert item["segments"] == 2
    assert library_status(cfg)["status"] == "transcribed"

    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    a, cut, end = data["segments"]["bounds_s"]
    assert a == 0 and 10 <= cut <= 13 and end == 23.0  # in the silence
    assert data["engine"]["model"] == "stt-async-v5"
    starts = [w["start_ms"] for w in data["words"]]
    assert len(data["words"]) == 2 * 7 and starts == sorted(starts)
    assert data["words"][7]["start_ms"] == round(cut * 1000)  # part 2 starts at the cut
    assert data["stats"]["words"] == 14
    assert len(data["source_tokens"]) == 2 * 12 and data["segments"]["source_token_ranges"] == [[0, 12], [12, 24]]
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert md.count("**Idea:** build a bird house") == 2 and "segments: 2" in md

    assert not (cfg.data_dir / "tmp" / LONG_ID).exists()  # cleaned up after success
    state = json.loads((cfg.data_dir / "state.json").read_text())[LONG_ID]
    assert state["ok"] and "segments" not in state
    jobs = [json.loads(p.read_text()) for p in (cfg.data_dir / "jobs").glob("*.json")]
    assert sorted(j["request"]["part"] for j in jobs if j["request"]["action"] == "transcribe_part") == [1, 2]
    assert all(j["status"] == "done" and j["failed"] == 0 for j in jobs)


@needs_ffmpeg
def test_parts_are_valid_audio(tmp_path, library):
    path = make_long_memo(library, tmp_path)
    out = segment.cut_part(path, 11.5, None, tmp_path / "part2.m4a")
    codec, duration = segment.probe(out)
    assert codec == "aac" and 11 <= duration <= 12.5
    assert path.stat().st_size > out.stat().st_size  # a copy of a part, the source untouched


@needs_ffmpeg
def test_only_the_failed_part_is_tried_again(tmp_path, library):
    make_long_memo(library, tmp_path)
    cfg = long_cfg(tmp_path, library)
    first = FakeTranscriber(fail_for=("part2",))
    summary = Worker(cfg, transcriber=first).run()
    assert "part2.m4a" in first.calls and "part1.m4a" in first.calls
    failed = next(i for i in summary["failed"] if i["short_id"] == "aaaa1111")
    assert failed["error"].startswith("part 2/2 failed")
    status = library_status(cfg)
    assert status["status"] == "segmenting" and status["segments"] == {"done": 1, "total": 2}
    workdir = cfg.data_dir / "tmp" / LONG_ID
    assert (workdir / "part1.tokens.json").exists()  # kept for the next run and for diagnosis

    second = FakeTranscriber()
    Worker(cfg, transcriber=second).run()
    assert second.calls == ["part2.m4a"]
    assert library_status(cfg)["status"] == "transcribed"
    assert not workdir.exists()


@needs_ffmpeg
def test_a_part_that_keeps_failing_fails_the_memo(tmp_path, library):
    make_long_memo(library, tmp_path)
    cfg = long_cfg(tmp_path, library, VOICEMEMOS_MAX_ATTEMPTS="2")
    for _ in range(3):
        Worker(cfg, transcriber=FakeTranscriber(fail_for=("part2",))).run()
    status = library_status(cfg)
    assert status["status"] == "failed" and "simulated failure" in status["error"]
    calls = FakeTranscriber()
    Worker(cfg, transcriber=calls).run()
    assert calls.calls == []  # no more automatic attempts


@needs_ffmpeg
def test_dry_run_shows_the_plan_without_calling_the_engine(tmp_path, library):
    make_long_memo(library, tmp_path)
    cfg = long_cfg(tmp_path, library)
    fake = FakeTranscriber()
    summary = Worker(cfg, transcriber=fake).run(dry_run=True, only=["aaaa1111"])
    assert fake.calls == []
    assert "2 parts, cut at 0:00:1" in summary["transcribed"][0]["text"]


def test_too_long_without_splitting_is_logged_and_listed(tmp_path, library, caplog):
    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZDURATION=? WHERE ZUNIQUEID=?", (30040.8, LONG_ID))
    db.commit()
    db.close()
    cfg = long_cfg(tmp_path, library, VOICEMEMOS_MAX_DURATION_MIN="298", VOICEMEMOS_SEGMENT_LONG="0")
    fake = FakeTranscriber()
    with caplog.at_level(logging.WARNING, logger="voicememos_mcp.worker"):
        Worker(cfg, transcriber=fake).run()
    assert fake.calls == ["20261014 090000-BBBB2222.qta"]
    assert any(r.levelno == logging.WARNING and LONG_ID in r.getMessage() for r in caplog.records)

    worker = json.loads((cfg.data_dir / "worker.json").read_text())
    assert [s["short_id"] for s in worker["skipped"]] == ["aaaa1111"]
    assert "VOICEMEMOS_SEGMENT_LONG=0" in worker["skipped"][0]["error"]
    status = library_status(cfg)
    assert status["status"] == "too_long" and "longer than" in status["error"]
    assert LONG_ID not in json.loads((cfg.data_dir / "state.json").read_text())  # nothing terminal stored

    from voicememos_mcp.tools import Tools

    assert any("will not be transcribed" in w for w in Tools(cfg).status()["warnings"])


def test_missing_ffmpeg_is_reported_at_the_start(tmp_path, library, monkeypatch, caplog):
    import voicememos_mcp.worker as worker_module

    monkeypatch.setattr(worker_module, "find_tool", lambda name: None)
    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZDURATION=? WHERE ZUNIQUEID=?", (30.0, LONG_ID))
    db.commit()
    db.close()
    cfg = long_cfg(tmp_path, library)
    with caplog.at_level(logging.ERROR, logger="voicememos_mcp.worker"):
        summary = Worker(cfg, transcriber=FakeTranscriber()).run()
    assert any("cannot be transcribed" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)
    assert "ffmpeg" in summary["warnings"][0]
    assert library_status(cfg)["status"] == "too_long"
    worker = json.loads((cfg.data_dir / "worker.json").read_text())
    assert worker["warnings"] and worker["skipped"][0]["short_id"] == "aaaa1111"


def test_backfill_of_a_memo_that_was_too_long(tmp_path, library):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg and ffprobe are needed")
    make_long_memo(library, tmp_path)
    off = long_cfg(tmp_path, library, VOICEMEMOS_SEGMENT_LONG="0")
    Worker(off, transcriber=FakeTranscriber()).run()
    assert library_status(off)["status"] == "too_long"
    on = long_cfg(tmp_path, library)  # e.g. after the update
    fake = FakeTranscriber()
    Worker(on, transcriber=fake).run()
    assert sorted(fake.calls) == ["part1.m4a", "part2.m4a"]
    assert library_status(on)["status"] == "transcribed"


# ----------------------------------------------------------------------------- voiceprint per part

def test_voiceprint_is_applied_in_every_part(cfg):
    np = pytest.importorskip("numpy")
    from tests.test_voiceprint import OTHER, OWNER, fake_embed, tone, words_for
    from voicememos_mcp import voiceprint

    model = cfg.get("VOICEMEMOS_VOICEPRINT_MODEL")
    vp = voiceprint.enroll("Alex", model, {f"m{i}": tone(OWNER + i * 3, 20, seed=i) for i in range(4)}, fake_embed)
    worker = Worker(cfg, embedder=fake_embed)
    worker.store.ensure_dirs()
    voiceprint.Voiceprint(worker.store.voiceprint_path).save(vp)
    # part 1 (0-16 s): owner, then someone else; part 2 (16-32 s): someone else, then the owner
    audio = np.concatenate([tone(OWNER, 8, 1), tone(OTHER, 8, 2), tone(OTHER, 8, 3), tone(OWNER, 8, 4)])
    worker._audio_loader = lambda path: audio
    words = words_for([("1:1", 0, 8), ("1:2", 8, 16), ("2:1", 16, 24), ("2:2", 24, 32)])
    speakers, summary = worker._identify_parts(cfg.data_dir / "x.m4a", words, [0, 16, 32])
    assert {k: v["label"] for k, v in speakers.items()} == {"1:1": "Alex", "1:2": "S2a", "2:1": "S1b", "2:2": "Alex"}
    assert summary["result"] == "match" and [p["speaker"] for p in summary["parts"]] == ["1:1", "2:2"]
