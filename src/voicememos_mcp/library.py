"""Read-only access to the Voice Memos library (``CloudRecordings.db``).

The database is a Core Data store. Column names differ slightly between macOS
versions, so every query is built from ``PRAGMA table_info`` and missing columns
are read as NULL. Nothing in the Voice Memos container is ever modified.

Notes on the schema (observed on macOS 26):

* ``ZCLOUDRECORDING`` holds one row per memo, including memos whose audio has not
  been downloaded from iCloud yet (``ZPATH`` set, file absent, ``ZLOCALDURATION`` 0).
* ``ZENCRYPTEDTITLE`` is the visible title (despite the name it is plain text);
  ``ZCUSTOMLABEL`` holds an ISO timestamp on recent versions.
* ``ZEVICTIONDATE`` is set for memos in "Recently Deleted".
* ``ZFOLDER`` points to ``ZFOLDER.Z_PK``; the folder name is ``ZENCRYPTEDNAME``.
* ``ZDATE`` is seconds since 2001-01-01 UTC (Core Data reference date).
* New recordings are stored as ``.qta`` (QuickTime audio), older ones as ``.m4a``.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

APPLE_EPOCH = dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc)
AUDIO_EXTENSIONS = {".m4a", ".qta", ".caf", ".mov", ".mp4", ".aac", ".wav", ".mp3", ".aiff"}
FILENAME_TIMESTAMP = re.compile(r"(\d{8})[ _](\d{6})")

TITLE_COLUMNS = ("ZENCRYPTEDTITLE", "ZCUSTOMLABELFORSORTING", "ZCUSTOMLABEL")


class LibraryError(RuntimeError):
    """The library cannot be read (missing folder, missing permission, broken database)."""


@dataclass
class Memo:
    id: str  # stable identifier (ZUNIQUEID), falls back to the file stem
    title: str | None
    recorded: str | None  # ISO 8601, local time with offset
    duration_s: float | None
    filename: str | None
    local: bool  # audio file present on this Mac
    deleted: bool  # in "Recently Deleted"
    folder: str | None

    @property
    def short_id(self) -> str:
        return self.id.replace("-", "")[:8].lower()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["short_id"] = self.short_id
        return data


def core_data_to_local(value) -> dt.datetime | None:
    try:
        return (APPLE_EPOCH + dt.timedelta(seconds=float(value))).astimezone()
    except (TypeError, ValueError, OverflowError):
        return None


def timestamp_from_filename(name: str) -> dt.datetime | None:
    """Voice Memos file names start with the local recording time: ``YYYYMMDD HHMMSS-XXXX``."""
    match = FILENAME_TIMESTAMP.search(name)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").astimezone()
    except ValueError:
        return None


def _connect_read_only(db_path: Path) -> tuple[sqlite3.Connection, str | None]:
    """Open the database read-only. If that fails (locked, WAL quirks), read a temporary copy."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        con.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        return con, None
    except sqlite3.Error:
        tmp = tempfile.mkdtemp(prefix="voicememos-db-")
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(db_path) + suffix)
            if src.exists():
                shutil.copy2(src, Path(tmp) / (db_path.name + suffix))
        con = sqlite3.connect(Path(tmp) / db_path.name, timeout=5)
        return con, tmp


def read_library(recordings_dir: Path) -> list[Memo]:
    """Return every memo in the library, newest first. Raises LibraryError if unreadable."""
    db_path = recordings_dir / "CloudRecordings.db"
    try:
        if not db_path.exists():
            if not recordings_dir.exists():
                raise LibraryError(
                    f"Voice Memos folder not found: {recordings_dir}. Open Voice Memos on this Mac once "
                    "and enable iCloud sync for it."
                )
            raise LibraryError(f"CloudRecordings.db not found in {recordings_dir}")
    except PermissionError as exc:
        raise LibraryError(
            f"No permission to read {recordings_dir}. Grant Full Disk Access to the process that runs "
            "this code (see README)."
        ) from exc

    try:
        con, tmp = _connect_read_only(db_path)
    except (sqlite3.Error, PermissionError, OSError) as exc:
        raise LibraryError(f"Cannot open {db_path}: {exc}") from exc

    try:
        columns = {row[1].upper() for row in con.execute("PRAGMA table_info(ZCLOUDRECORDING)")}
        if not columns:
            raise LibraryError("Table ZCLOUDRECORDING not found; unsupported Voice Memos version")

        def col(name: str) -> str:
            return f"r.{name}" if name in columns else "NULL"

        title_expr = next((f"r.{name}" for name in TITLE_COLUMNS if name in columns), "NULL")
        folder_join, folder_expr = "", "NULL"
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ZFOLDER" in columns and "ZFOLDER" in tables:
            folder_columns = {row[1].upper() for row in con.execute("PRAGMA table_info(ZFOLDER)")}
            if "ZENCRYPTEDNAME" in folder_columns:
                folder_join = "LEFT JOIN ZFOLDER f ON f.Z_PK = r.ZFOLDER"
                folder_expr = "f.ZENCRYPTEDNAME"

        query = (
            f"SELECT {col('ZUNIQUEID')}, {title_expr}, {col('ZDATE')}, {col('ZDURATION')}, "
            f"{col('ZPATH')}, {col('ZEVICTIONDATE')}, {folder_expr} "
            f"FROM ZCLOUDRECORDING r {folder_join}"
        )
        memos: list[Memo] = []
        for unique_id, title, zdate, duration, zpath, eviction, folder in con.execute(query):
            filename = Path(zpath).name if zpath else None
            recorded = core_data_to_local(zdate) if zdate is not None else None
            if recorded is None and filename:
                recorded = timestamp_from_filename(filename)
            memo_id = unique_id or (Path(filename).stem if filename else None)
            if not memo_id:
                continue
            memos.append(
                Memo(
                    id=str(memo_id),
                    title=title.strip() if isinstance(title, str) and title.strip() else None,
                    recorded=recorded.isoformat(timespec="seconds") if recorded else None,
                    duration_s=float(duration) if duration else None,
                    filename=filename,
                    local=bool(filename and (recordings_dir / filename).exists()),
                    deleted=eviction is not None,
                    folder=folder.strip() if isinstance(folder, str) and folder.strip() else None,
                )
            )
    except sqlite3.Error as exc:
        raise LibraryError(f"Cannot read {db_path}: {exc}") from exc
    finally:
        con.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    memos.sort(key=lambda m: m.recorded or "", reverse=True)
    return memos


def is_excluded(memo: Memo, title_regex: str, folders: set[str]) -> bool:
    """True when the memo matches a configured exclusion. Excluded memos are hidden everywhere."""
    if folders and memo.folder and memo.folder.casefold() in folders:
        return True
    if title_regex and memo.title and re.search(title_regex, memo.title, re.IGNORECASE):
        return True
    return False
