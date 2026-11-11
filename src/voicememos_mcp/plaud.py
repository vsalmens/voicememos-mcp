"""Fetch recordings of a Plaud recorder (Note, Note Pro, NotePin) into the import folder.

A Plaud Note Pro has no USB file access: its recordings reach the phone app over Bluetooth or
Wi-Fi and from there Plaud's cloud, and the original audio is available only from the cloud.
This module reads it through Plaud's developer API, the one Plaud's own command-line tool
(``npm i -g @plaud-ai/cli``) uses:

* ``plaud login`` (run once, by the user, in a browser) stores OAuth tokens in
  ``~/.plaud/tokens.json``. This module reads that file and refreshes the access token the way
  the CLI does, writing the file back in the same shape, so both keep working.
* ``GET /open/third-party/files/`` lists recordings, ``GET /open/third-party/files/<id>`` gives
  one with ``presigned_url``, a link to the audio that is valid for 24 hours.

Each new recording is saved as ``plaud_<id>.<ext>`` in the import folder with the sidecar the
import command writes (title, recording time, duration, folder "Plaud"); the worker then treats
it like any imported recording. Only audio and its metadata are fetched: Plaud's own transcripts
and summaries are not read. ``DATA_DIR/plaud.json`` remembers what was fetched, so a recording
whose file is deleted from the import folder is not fetched again.

The worker asks every minute, so an idle call is kept small: most calls list only the last
few days (``RECENT_DAYS`` before the newest recording fetched), a full listing is made once an
hour, a recording that fails is tried again after a growing pause (``FAILURE_BACKOFF_S``), and
after a 429 nothing is asked until the time Plaud names. ``plaud.json`` is written only when
something in it changes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from .store import now_iso, read_json, write_json_atomic

log = logging.getLogger(__name__)

API_BASE = "https://platform.plaud.ai/developer/api"
REFRESH_PATH = "/oauth/third-party/access-token/refresh"
PLAUD_FOLDER = "Plaud"
PAGE_SIZE = 100
MAX_PAGES = 1000
RECENT_DAYS = 3  # a recording reaches the cloud, and gets its AI title, well within this
FULL_LISTING_EVERY_S = 3600  # catches what the short listing cannot: old recordings synced late
FAILURE_BACKOFF_S = (300, 3600, 6 * 3600, 86400)  # per recording: after the 1st, 2nd, 3rd failure, then daily
RATE_LIMIT_BACKOFF_S = (300, 900, 3600)  # after a 429 without Retry-After
REFRESH_MARGIN_MS = 60_000
# Containers Soniox reads as they are (audio.DIRECT_EXTENSIONS) that a recorder may produce.
KNOWN_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".mp4"}
CONTENT_TYPES = {"mpeg": ".mp3", "mp3": ".mp3", "ogg": ".ogg", "opus": ".ogg", "wav": ".wav", "wave": ".wav",
                 "mp4": ".m4a", "m4a": ".m4a", "aac": ".aac"}
_HAS_ZONE = re.compile(r"(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


class PlaudError(RuntimeError):
    pass


class PlaudAuthError(PlaudError):
    """No usable tokens: the user has to run ``plaud login`` again."""


class PlaudRateLimited(PlaudError):
    """429: stop asking for a while (``retry_after`` seconds when the answer says how long)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _epoch(iso) -> float:
    try:
        return dt.datetime.fromisoformat(iso).timestamp() if iso else 0.0
    except (TypeError, ValueError):
        return 0.0


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def parse_time(value) -> dt.datetime | None:
    """A time from the API as an aware local datetime. A string without a zone is UTC (as in
    Plaud's own client); a number is seconds or milliseconds since 1970."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value
        try:
            return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).astimezone()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if text.isdigit():
        return parse_time(int(text))
    if not _HAS_ZONE.search(text):
        text += "+00:00"
    elif text[-1] in "Zz":
        text = text[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(text.replace(" ", "T", 1)).astimezone()
    except ValueError:
        return None


def sniff_extension(head: bytes) -> str | None:
    """The container by a file's first bytes, for links and responses that name no type."""
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"RIFF":
        return ".wav"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return ".mp3"
    return None


def extension_for(url: str, content_type: str, head: bytes) -> str | None:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".opus":
        suffix = ".ogg"  # Opus in an Ogg container; the pipeline knows the container's name
    if suffix in KNOWN_EXTENSIONS:
        return suffix
    kind = content_type.split(";")[0].strip().lower().rpartition("/")[2].removeprefix("x-")
    return CONTENT_TYPES.get(kind) or sniff_extension(head)


class Plaud:
    """Plaud's developer API with the tokens ``plaud login`` stored."""

    def __init__(self, tokens_path: Path, api_base: str = API_BASE, region: str = "", session=None,
                 clock=time.time, timeout: float = 60.0):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.tokens_path = tokens_path
        self.base = (api_base or API_BASE).rstrip("/")
        self.extra = {"x-pld-region": region} if region else {}
        self.clock = clock
        self.timeout = timeout

    # ---- tokens
    def _load_tokens(self) -> dict:
        try:
            tokens = json.loads(self.tokens_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise PlaudAuthError(f"no Plaud login found ({self.tokens_path}); run `plaud login`") from None
        except (OSError, ValueError) as exc:
            raise PlaudAuthError(f"cannot read {self.tokens_path}: {exc}") from exc
        if not isinstance(tokens, dict) or not tokens.get("access_token"):
            raise PlaudAuthError(f"{self.tokens_path} holds no access token; run `plaud login`")
        return tokens

    def _save_tokens(self, tokens: dict) -> None:
        """Same shape and mode (600) as the CLI writes, replaced atomically."""
        tmp = self.tokens_path.with_name(f".{self.tokens_path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(tokens, handle, indent=2)
        os.replace(tmp, self.tokens_path)

    def _refresh(self, tokens: dict) -> dict:
        if not tokens.get("refresh_token"):
            raise PlaudAuthError("the Plaud access token has expired and there is no refresh token; run `plaud login`")
        response = self.session.post(
            self.base + REFRESH_PATH, data={"refresh_token": tokens["refresh_token"]},
            headers={"Accept": "application/json", **self.extra}, timeout=self.timeout)
        if response.status_code in (401, 403):
            raise PlaudAuthError(f"Plaud refused the stored login ({response.status_code}); run `plaud login`")
        if not response.ok:
            raise PlaudError(f"Plaud token refresh failed: {response.status_code} {response.text[:200]}")
        data = response.json()
        if not data.get("access_token"):
            raise PlaudError("Plaud token refresh returned no access token")
        fresh = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token") or tokens["refresh_token"],
            "token_type": data.get("token_type") or "Bearer",
        }
        if data.get("expires_in"):
            fresh["expires_at"] = int(self.clock() * 1000 + float(data["expires_in"]) * 1000)
        self._save_tokens(fresh)
        return fresh

    def _access_token(self, force_refresh: bool = False) -> str:
        tokens = self._load_tokens()
        expires = tokens.get("expires_at")
        if force_refresh or (expires and self.clock() * 1000 > float(expires) - REFRESH_MARGIN_MS):
            tokens = self._refresh(tokens)
        return tokens["access_token"]

    # ---- API
    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in (0, 1):
            token = self._access_token(force_refresh=attempt == 1)
            response = self.session.get(
                self.base + path, params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json", **self.extra},
                timeout=self.timeout)
            if response.status_code == 401 and attempt == 0:
                continue  # a token without expires_at, or one revoked early: refresh once
            if response.status_code in (401, 403):
                raise PlaudAuthError(f"Plaud refused the request ({response.status_code}); run `plaud login`")
            if response.status_code == 429:
                wait = str((getattr(response, "headers", None) or {}).get("Retry-After") or "").strip()
                raise PlaudRateLimited(f"Plaud 429 for {path}", float(wait) if wait.isdigit() else None)
            if not response.ok:
                raise PlaudError(f"Plaud {response.status_code} for {path}: {response.text[:200]}")
            return response.json()
        raise PlaudError("unreachable")

    def list_files(self, since: dt.datetime | None = None) -> list[dict]:
        """Every recording of the account (from ``since`` on), as the API lists them."""
        files: list[dict] = []
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            params = {"page": page, "page_size": PAGE_SIZE}
            if since is not None:
                params["start_date"] = int(since.timestamp() * 1000)
            batch = self._get("/open/third-party/files/", params).get("data") or []
            fresh = [f for f in batch if isinstance(f, dict) and f.get("id") and str(f["id"]) not in seen]
            # Not "a short page": a server that caps the page size below PAGE_SIZE would then
            # show its first page only. An empty page ends the list, and so does a page with
            # nothing new (a server that ignores the page number).
            if not fresh:
                break
            seen.update(str(f["id"]) for f in fresh)
            files += fresh
        return files

    def get_file(self, file_id: str) -> dict:
        return self._get(f"/open/third-party/files/{file_id}")

    def download(self, url: str, dest_stem: Path) -> Path:
        """Stream the audio behind a presigned link to ``dest_stem`` + the right extension.
        The link carries its own signature, so no login header is sent with it."""
        if urlparse(url).scheme != "https":
            raise PlaudError("the audio link is not https")
        part = dest_stem.with_name(f".{dest_stem.name}.part")  # dot files are not listed as imports
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as response:
                if not response.ok:
                    raise PlaudError(f"audio download failed: {response.status_code}")
                head = b""
                with open(part, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            if len(head) < 16:
                                head += chunk[:16]
                            handle.write(chunk)
                suffix = extension_for(url, response.headers.get("Content-Type", ""), head)
            if part.stat().st_size == 0:
                raise PlaudError("the audio file is empty")
            if not suffix:
                raise PlaudError("unknown audio format (neither the link, the content type nor the file names one)")
            dest = dest_stem.with_name(dest_stem.name + suffix)
            os.replace(part, dest)
            return dest
        finally:
            part.unlink(missing_ok=True)


def _start(file: dict) -> dt.datetime | None:
    return parse_time(file.get("start_at")) or parse_time(file.get("created_at"))


def _sidecar(file: dict) -> dict:
    meta: dict = {"folder": PLAUD_FOLDER}
    name = str(file.get("name") or "").strip()
    if name:
        meta["title"] = name
    started = _start(file)
    if started:
        meta["recorded"] = started.isoformat(timespec="seconds")
    duration = file.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        meta["duration_s"] = round(duration / 1000, 3)
    return meta


def _write_sidecar(audio: Path, meta: dict) -> None:
    audio.with_name(audio.name + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _rate_limit(state: dict, now: float, exc: PlaudRateLimited) -> str:
    count = int((state.get("rate_limited") or {}).get("count", 0)) + 1
    wait = max(exc.retry_after or 0, RATE_LIMIT_BACKOFF_S[min(count, len(RATE_LIMIT_BACKOFF_S)) - 1])
    state["rate_limited"] = {"until": _iso(now + wait), "count": count}
    log.warning("Plaud asked for fewer requests (429); next try at %s", state["rate_limited"]["until"])
    return state["rate_limited"]["until"]


def fetch_new(client: Plaud, import_dir: Path, state_path: Path, since: str = "", limit: int | None = None,
              dry_run: bool = False, clock=time.time) -> dict:
    """Download the recordings not fetched before. Returns a summary without titles (it is logged).

    One recording failing does not stop the others; it is tried again after a pause that grows
    with every failure. A failed login stops the run (``PlaudAuthError``). A recording whose
    audio is not in the cloud yet (still on the device or the phone) has no link and is tried
    again on the next run. A recording renamed in Plaud after it was fetched gets the new title
    in its sidecar; the worker carries it into the transcript.
    """
    state = read_json(state_path, {}) or {}
    known: dict = state.setdefault("files", {})
    failures: dict = state.setdefault("failures", {})
    before = json.dumps(state, sort_keys=True)
    now = clock()
    summary: dict = {"listed": 0, "fetched": [], "waiting": [], "failed": [], "renamed": 0, "failing": len(failures)}

    def save() -> None:
        if not dry_run and json.dumps(state, sort_keys=True) != before:
            state["checked"] = _iso(now)
            write_json_atomic(state_path, state)

    until = (state.get("rate_limited") or {}).get("until")
    if not dry_run and _epoch(until) > now:
        summary["rate_limited"] = until
        return summary
    since_dt = None
    if since:
        try:
            since_dt = dt.datetime.fromisoformat(since).astimezone()
        except ValueError:
            raise PlaudError(f"VOICEMEMOS_PLAUD_SINCE must be YYYY-MM-DD, got {since!r}") from None
    # Most runs ask only for the days around the newest recording fetched; once an hour for all.
    starts = [_epoch(e.get("start")) for e in known.values() if e.get("start")]
    full = dry_run or not starts or now - _epoch(state.get("listed_all")) >= FULL_LISTING_EVERY_S
    list_since = since_dt
    if not full:
        recent = dt.datetime.fromtimestamp(max(starts) - RECENT_DAYS * 86400).astimezone()
        list_since = max(since_dt, recent) if since_dt else recent
    try:
        files = client.list_files(list_since)
    except PlaudRateLimited as exc:
        summary["rate_limited"] = _rate_limit(state, now, exc)
        save()
        return summary
    state.pop("rate_limited", None)
    files.sort(key=lambda f: (_start(f) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)))
    summary["listed"] = len(files)
    if not dry_run:
        import_dir.mkdir(parents=True, exist_ok=True)
    if full and not dry_run:
        state["listed_all"] = _iso(now)
        listed = {str(f["id"]) for f in files}
        for gone in [k for k in failures if k not in listed]:  # deleted in Plaud: nothing to retry
            del failures[gone]
    for file in files:
        file_id = str(file["id"])
        started = _start(file)
        if since_dt and started and started < since_dt:
            continue  # the server-side filter is not applied for every account
        entry = known.get(file_id)
        if entry:
            if started and not entry.get("start"):
                entry["start"] = started.isoformat(timespec="seconds")
            name = str(file.get("name") or "").strip()
            audio = import_dir / str(entry.get("filename") or "")
            if name and name != entry.get("name") and entry.get("filename") and audio.is_file() and not dry_run:
                _write_sidecar(audio, {**_kept(audio), **_sidecar(file)})
                entry["name"] = name
                summary["renamed"] += 1
            continue
        if limit is not None and len(summary["fetched"]) >= limit:
            break
        if dry_run:
            summary["fetched"].append(file_id)
            continue
        failure = failures.get(file_id) or {}
        if _epoch(failure.get("next_try")) > now:
            continue  # failed a while ago; its pause is not over
        try:
            detail = client.get_file(file_id)
            url = detail.get("presigned_url")
            if not url:
                summary["waiting"].append(file_id)
                continue
            merged = {**file, **{k: v for k, v in detail.items() if v not in (None, "")}}
            audio = client.download(url, import_dir / f"plaud_{_SAFE_ID.sub('', file_id)}")
            _write_sidecar(audio, _sidecar(merged))
            when = _start(merged)
            if when:  # the recording's own time: old enough for the worker to take it in this run
                os.utime(audio, (when.timestamp(), when.timestamp()))
            known[file_id] = {"filename": audio.name, "name": str(merged.get("name") or "").strip(),
                              "fetched": now_iso(), **({"start": when.isoformat(timespec="seconds")} if when else {})}
            failures.pop(file_id, None)
            summary["fetched"].append(file_id)
            write_json_atomic(state_path, state)
        except PlaudAuthError:
            save()
            raise
        except PlaudRateLimited as exc:
            summary["rate_limited"] = _rate_limit(state, now, exc)
            break
        except Exception as exc:  # noqa: BLE001 - one recording must not stop the others
            count = int(failure.get("count", 0)) + 1
            pause = FAILURE_BACKOFF_S[min(count, len(FAILURE_BACKOFF_S)) - 1]
            failures[file_id] = {"count": count, "error": str(exc)[:200], "at": _iso(now),
                                 "next_try": _iso(now + pause)}
            log.warning("Plaud recording %s not fetched (try %d, next in %d min): %s", file_id, count, pause // 60, exc)
            summary["failed"].append({"id": file_id, "error": str(exc)[:200], "tries": count})
    summary["failing"] = len(failures)
    save()
    return summary


def _kept(audio: Path) -> dict:
    """What the worker added to a sidecar (a probed duration) survives a rename."""
    try:
        data = json.loads(audio.with_name(audio.name + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k in ("duration_s", "duration_probed")} if isinstance(data, dict) else {}


def client_from_config(cfg, session=None) -> Plaud:
    return Plaud(cfg.plaud_tokens_path, cfg.get("VOICEMEMOS_PLAUD_API_BASE"), cfg.get("VOICEMEMOS_PLAUD_REGION"),
                 session=session)
