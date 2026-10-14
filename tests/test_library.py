import shutil
import subprocess
from pathlib import Path

import pytest

from voicememos_mcp.library import LibraryError, is_excluded, read_library


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
