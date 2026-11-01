import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from voicememos_mcp.library import LibraryError, import_id, is_excluded, read_imports, read_library


def test_imports_folder_gives_memos(tmp_path):
    folder = tmp_path / "imports"
    folder.mkdir()
    (folder / "GMT20261031-102304_Recording.m4a").write_bytes(b"\0" * 16)
    (folder / "talk.m4a").write_bytes(b"\0" * 16)
    (folder / "talk.m4a.json").write_text(json.dumps(
        {"title": "Board meeting", "recorded": "2026-10-31T13:00:00+02:00", "duration_s": 8213.3, "folder": "Work"}))
    (folder / "notes.txt").write_text("not audio")
    (folder / ".DS_Store").write_bytes(b"")

    memos = read_imports(folder, probe=lambda p: 60.0)
    assert [m.filename for m in memos] == ["talk.m4a", "GMT20261031-102304_Recording.m4a"]  # newest first
    zoom, talk = memos[1], memos[0]
    # Zoom names carry the UTC time
    assert dt.datetime.fromisoformat(zoom.recorded) == dt.datetime(2026, 10, 31, 10, 23, 4, tzinfo=dt.timezone.utc)
    assert zoom.title == "GMT20261031-102304_Recording" and zoom.duration_s == 60.0 and zoom.folder == "Imported"
    assert zoom.local and not zoom.deleted and zoom.imported and zoom.path == str(folder / zoom.filename)
    assert zoom.id == import_id(zoom.filename) and len(zoom.id) == 32 and len(zoom.short_id) == 8
    assert talk.title == "Board meeting" and talk.duration_s == 8213.3 and talk.folder == "Work"
    assert talk.recorded == "2026-10-31T13:00:00+02:00"
    assert read_imports(tmp_path / "missing") == []


def test_reads_memos_newest_first(library):
    memos = read_library(library)
    assert len(memos) == 5
    assert memos[0].title == "Car notes"
    by_title = {m.title: m for m in memos}
    assert by_title["Old memo in iCloud"].local is False
    assert by_title["Deleted one"].deleted is True
    assert by_title["Secret plans"].folder == "Private"
    assert by_title["Morning walk"].short_id == "aaaa1111"
    assert by_title["Morning walk"].recorded[10] == "T"


def test_exclusion_by_folder_and_title(library):
    memos = {m.title: m for m in read_library(library)}
    assert is_excluded(memos["Secret plans"], "", {"private"})
    assert is_excluded(memos["Car notes"], "^car", set())
    assert not is_excluded(memos["Morning walk"], "^car", {"private"})


def test_missing_folder_raises(tmp_path):
    with pytest.raises(LibraryError):
        read_library(tmp_path / "nope")


def test_tolerates_missing_columns(tmp_path):
    import sqlite3

    rec = tmp_path / "R"
    rec.mkdir()
    con = sqlite3.connect(rec / "CloudRecordings.db")
    con.execute("CREATE TABLE ZCLOUDRECORDING (Z_PK INTEGER, ZPATH VARCHAR, ZCUSTOMLABEL VARCHAR)")
    con.execute("INSERT INTO ZCLOUDRECORDING VALUES (1, '20200101 120000-X.m4a', 'Label')")
    con.commit()
    memos = read_library(rec)
    assert memos[0].id == "20200101 120000-X" and memos[0].title == "Label"
    assert memos[0].recorded.startswith("2020-01-01T12:00:00")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_qta_conversion_copies_audio_track(tmp_path):
    from voicememos_mcp.audio import prepare_audio

    src = tmp_path / "x.mov"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:a", "aac", str(src)], check=True)
    qta = src.with_suffix(".qta")
    src.rename(qta)
    out = prepare_audio(qta, tmp_path)
    assert out.suffix == ".m4a" and out.stat().st_size > 0
    assert prepare_audio(out, tmp_path) == out  # m4a passes through untouched


def test_unsupported_format(tmp_path):
    from voicememos_mcp.audio import prepare_audio

    f = tmp_path / "x.xyz"
    f.write_bytes(b"0")
    with pytest.raises(RuntimeError):
        prepare_audio(f, Path(tmp_path))
