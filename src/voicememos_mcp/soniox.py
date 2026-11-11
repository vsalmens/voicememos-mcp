"""Minimal client for the Soniox async speech-to-text API.

Flow: upload file -> create transcription -> poll -> fetch tokens -> delete both
the transcription and the uploaded file. Deletion runs on every exit path so no
audio or transcript is left on Soniox's servers. A process that is killed outright
skips that path, so with ``pending_path`` the ids are written to a small file when they
are created and removed when they are deleted; ``cleanup_leftovers`` deletes what an
earlier run left there.

A lost connection is retried (while polling, for as long as the poll may take: the
transcription goes on at Soniox meanwhile). When the service cannot be reached or refuses
the key, ``SonioxUnavailable`` is raised: that says nothing about the recording, so the
worker does not count it as a failed attempt of the memo.

Each returned token carries ``text``, ``start_ms``, ``end_ms``, ``confidence``
(0..1), ``language`` (language identification) and, with diarization enabled,
``speaker`` (a label that is only meaningful within one recording).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
REFUSED_STATUS = {401, 402, 403}  # the key or the account, not the recording
POLL_RETRY_MAX_S = 60.0


class SonioxError(RuntimeError):
    pass


class SonioxUnavailable(SonioxError):
    """No answer, an overloaded service or a refused key: nothing wrong with the recording."""


class SonioxRefused(SonioxUnavailable):
    """The key or the account was refused: waiting does not help, a person has to look."""


class Soniox:
    def __init__(self, api_key: str, base_url: str = "https://api.soniox.com",
                 model: str = "stt-async-v6", session=None, poll_interval: float = 2.0,
                 poll_timeout: float = 1800.0, retries: int = 3, pending_path: Path | None = None,
                 sleep=time.sleep):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.base = base_url.rstrip("/")
        self.model = model
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.retries = retries
        self.pending_path = pending_path
        self.sleep = sleep

    # ---- what is at Soniox right now (survives a killed process)
    def _pending(self) -> list[str]:
        if self.pending_path is None:
            return []
        try:
            paths = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [p for p in paths if isinstance(p, str)] if isinstance(paths, list) else []

    def _save_pending(self, paths: list[str]) -> None:
        if self.pending_path is None:
            return
        try:
            if not paths:
                self.pending_path.unlink(missing_ok=True)
                return
            tmp = self.pending_path.with_name(f".{self.pending_path.name}.tmp")
            tmp.write_text(json.dumps(paths), encoding="utf-8")
            os.replace(tmp, self.pending_path)
        except OSError as exc:  # bookkeeping must not stop a transcription
            log.warning("Cannot write %s: %s", self.pending_path, exc)

    def _remember(self, path: str) -> None:
        self._save_pending(self._pending() + [path])

    def _forget(self, path: str) -> None:
        pending = self._pending()
        if path in pending:
            self._save_pending([p for p in pending if p != path])

    def _delete(self, path: str) -> bool:
        """Delete a transcription or a file. True when it is gone (also when it never existed)."""
        try:
            response = self.session.delete(f"{self.base}{path}", timeout=60)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
            log.warning("Soniox cleanup failed for %s: %s", path, exc)
            return False
        if not response.ok and response.status_code != 404:
            log.warning("Soniox cleanup failed for %s: %s", path, response.status_code)
            return False
        return True

    def cleanup_leftovers(self) -> int:
        """Delete what a killed run left at Soniox (ids from ``pending_path``). Returns how many."""
        removed = 0
        for path in self._pending():
            if self._delete(path):
                self._forget(path)
                removed += 1
        return removed

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        delay = 2.0
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                response = self.session.request(method, url, **kwargs)
            except OSError as exc:  # requests' connection errors and timeouts are OSErrors
                if last:
                    raise SonioxUnavailable(f"Soniox cannot be reached: {str(exc)[:200]}") from exc
                log.warning("Soniox %s %s: %s, retrying in %.0fs", method, path, type(exc).__name__, delay)
            else:
                if response.status_code in RETRY_STATUS:
                    if last:
                        raise SonioxUnavailable(f"Soniox {response.status_code}: {response.text[:400]}")
                    log.warning("Soniox %s %s -> %s, retrying in %.0fs", method, path, response.status_code, delay)
                elif response.status_code in REFUSED_STATUS:
                    hint = ""
                    if response.status_code in (401, 403) and ".eu." in self.base:
                        hint = " (the EU endpoint needs a key from an EU project)"
                    raise SonioxRefused(f"Soniox {response.status_code}: {response.text[:400]}{hint}")
                elif not response.ok:
                    raise SonioxError(f"Soniox {response.status_code}: {response.text[:400]}")
                else:
                    return response
            self.sleep(delay)
            delay *= 2
            # Re-open file uploads on retry: the stream has been consumed.
            if "files" in kwargs and hasattr(kwargs["files"].get("file", (None, None))[1], "seek"):
                kwargs["files"]["file"][1].seek(0)
        raise SonioxError(f"Soniox: giving up on {method} {path}")

    def transcribe(self, audio: Path, language_hints: list[str], context_terms: list[str] | None = None,
                   context_text: str = "", reference: str = "voicememos-mcp",
                   diarization: bool = False) -> list[dict]:
        with open(audio, "rb") as fh:
            file_id = self._request("POST", "/v1/files", files={"file": (audio.name, fh)}, timeout=600).json()["id"]
        self._remember(f"/v1/files/{file_id}")
        transcription_id = None
        try:
            body: dict = {
                "model": self.model,
                "file_id": file_id,
                "language_hints": language_hints,
                "enable_language_identification": True,
                "client_reference_id": reference,
            }
            if diarization:
                body["enable_speaker_diarization"] = True
            context: dict = {}
            if context_terms:
                context["terms"] = context_terms
            if context_text:
                context["text"] = context_text
            if context:
                body["context"] = context
            transcription_id = self._request("POST", "/v1/transcriptions", json=body, timeout=60).json()["id"]
            self._remember(f"/v1/transcriptions/{transcription_id}")

            started = time.monotonic()
            wait = self.poll_interval
            while True:
                try:
                    status = self._request("GET", f"/v1/transcriptions/{transcription_id}", timeout=60).json()
                except SonioxRefused:
                    raise
                except SonioxUnavailable as exc:
                    # The network dropped or the Mac slept: the transcription goes on at Soniox,
                    # so keep asking instead of uploading (and paying for) the file again.
                    if time.monotonic() - started > self.poll_timeout:
                        raise
                    wait = min(max(wait * 2, self.poll_interval), POLL_RETRY_MAX_S)
                    log.warning("Soniox status not available (%s); asking again in %.0fs", exc, wait)
                    self.sleep(wait)
                    continue
                wait = self.poll_interval
                state = status.get("status")
                if state == "completed":
                    break
                if state == "error":
                    raise SonioxError(f"Soniox transcription failed: {status.get('error_message')}")
                if time.monotonic() - started > self.poll_timeout:
                    raise SonioxError("Soniox transcription timed out")
                self.sleep(self.poll_interval)

            transcript = self._request("GET", f"/v1/transcriptions/{transcription_id}/transcript", timeout=120)
            return transcript.json().get("tokens", [])
        finally:
            targets = ([f"/v1/transcriptions/{transcription_id}"] if transcription_id else []) + [f"/v1/files/{file_id}"]
            for path in targets:
                if self._delete(path):
                    self._forget(path)  # what could not be deleted now is deleted by the next run
