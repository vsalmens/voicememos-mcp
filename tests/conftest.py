import os
import sqlite3
import time
from pathlib import Path

import pytest

from voicememos_mcp.config import load_config

# Columns as observed in CloudRecordings.db on macOS 26 (subset that matters + a few extras).
RECORDING_COLUMNS = [
    ("Z_PK", "INTEGER PRIMARY KEY"), ("Z_ENT", "INTEGER"), ("Z_OPT", "INTEGER"), ("ZFLAGS", "INTEGER"),
    ("ZFOLDER", "INTEGER"), ("ZDATE", "TIMESTAMP"), ("ZDURATION", "FLOAT"), ("ZEVICTIONDATE", "TIMESTAMP"),
    ("ZLOCALDURATION", "FLOAT"), ("ZCUSTOMLABEL", "VARCHAR"), ("ZCUSTOMLABELFORSORTING", "VARCHAR"),
    ("ZENCRYPTEDTITLE", "VARCHAR"), ("ZPATH", "VARCHAR"), ("ZUNIQUEID", "VARCHAR"),
]

APPLE_EPOCH_UNIX = 978307200  # 2001-01-01T00:00:00Z


def core_data(unix_ts: float) -> float:
    return unix_ts - APPLE_EPOCH_UNIX


# Sub-word tokens in the shape Soniox returns them.
TOKENS = [
    {"text": "Hello", "start_ms": 0, "end_ms": 300, "confidence": 0.99, "language": "en"},
    {"text": " wor", "start_ms": 320, "end_ms": 500, "confidence": 0.95, "language": "en"},
    {"text": "ld", "start_ms": 500, "end_ms": 650, "confidence": 0.40, "language": "en"},
    {"text": ".", "start_ms": 650, "end_ms": 660, "confidence": 0.20, "language": "en"},
    {"text": " Idea", "start_ms": 4000, "end_ms": 4300, "confidence": 0.98, "language": "en"},
    {"text": ":", "start_ms": 4300, "end_ms": 4310, "confidence": 0.99, "language": "en"},
    {"text": " build", "start_ms": 4400, "end_ms": 4700, "confidence": 0.97, "language": "en"},
    {"text": " a", "start_ms": 4700, "end_ms": 4750, "confidence": 0.96, "language": "en"},
    {"text": " bird", "start_ms": 4750, "end_ms": 5000, "confidence": 0.91, "language": "en"},
    {"text": " house", "start_ms": 5000, "end_ms": 5300, "confidence": 0.93, "language": "en"},
    {"text": ".", "start_ms": 5300, "end_ms": 5310, "confidence": 0.99, "language": "en"},
    {"text": "<end>", "start_ms": 5310, "end_ms": 5310, "confidence": 1.0},
]


@pytest.fixture
def library(tmp_path: Path):
    rec = tmp_path / "Recordings"
    rec.mkdir()
    db = sqlite3.connect(rec / "CloudRecordings.db")
    db.execute(f"CREATE TABLE ZCLOUDRECORDING ({', '.join(f'{n} {t}' for n, t in RECORDING_COLUMNS)})")
    db.execute("CREATE TABLE ZFOLDER (Z_PK INTEGER PRIMARY KEY, ZENCRYPTEDNAME VARCHAR, ZUUID VARCHAR)")
    db.execute("INSERT INTO ZFOLDER VALUES (1, 'Private', 'f-1')")
    now = time.time()
    rows = [
        # unique id, title, recorded (unix), duration, path, evicted, folder, local file?
        ("AAAA1111-0000-0000-0000-000000000001", "Morning walk", now - 86400 * 2, 12.0,
         "20261012 080000-AAAA1111.m4a", None, None, True),
        ("BBBB2222-0000-0000-0000-000000000002", "Car notes", now - 3600 * 3, 8.0,
         "20261014 090000-BBBB2222.qta", None, None, True),
        ("CCCC3333-0000-0000-0000-000000000003", "Old memo in iCloud", now - 86400 * 400, 30.0,
         "20250901 100000-CCCC3333.m4a", None, None, False),
        ("DDDD4444-0000-0000-0000-000000000004", "Deleted one", now - 86400, 5.0,
         "20261013 100000-DDDD4444.m4a", core_data(now), None, True),
        ("EEEE5555-0000-0000-0000-000000000005", "Secret plans", now - 86400, 5.0,
         "20261013 110000-EEEE5555.m4a", None, 1, True),
    ]
    for i, (uid, title, ts, dur, path, evicted, folder, local) in enumerate(rows, start=1):
        db.execute(
            "INSERT INTO ZCLOUDRECORDING (Z_PK, ZDATE, ZDURATION, ZEVICTIONDATE, ZFOLDER, ZENCRYPTEDTITLE, "
            "ZCUSTOMLABEL, ZPATH, ZUNIQUEID) VALUES (?,?,?,?,?,?,?,?,?)",
            (i, core_data(ts), dur, evicted, folder, title, "2026-10-14T06:00:00Z", path, uid),
        )
        if local:
            f = rec / path
            f.write_bytes(b"\0" * 64)
            old = now - 3600
            os.utime(f, (old, old))
    db.commit()
    db.close()
    return rec


@pytest.fixture
def cfg(tmp_path: Path, library: Path):
    env = {
        "VOICEMEMOS_RECORDINGS_DIR": str(library),
        "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
        "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"),
        "VOICEMEMOS_EXCLUDE_FOLDERS": "private",
        "VOICEMEMOS_WORKER": "none",
        "VOICEMEMOS_SINCE": "",
    }
    return load_config(path=tmp_path / "missing.env", env=env)


class FakeTranscriber:
    def __init__(self, tokens=None, fail_for=()):
        self.tokens = tokens or TOKENS
        self.fail_for = fail_for
        self.calls = []

    def __call__(self, audio: Path):
        self.calls.append(audio.name)
        if any(f in audio.name for f in self.fail_for):
            raise RuntimeError("simulated failure")
        return [dict(t) for t in self.tokens]


@pytest.fixture
def fake():
    return FakeTranscriber()


@pytest.fixture(autouse=True)
def no_real_audio_conversion(monkeypatch):
    """Tests use dummy files; skip ffmpeg/afconvert except in the dedicated audio test."""
    import voicememos_mcp.worker as worker

    monkeypatch.setattr(worker, "prepare_audio", lambda path, workdir: path)
