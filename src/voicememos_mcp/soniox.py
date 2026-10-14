"""Minimal client for the Soniox async speech-to-text API.

Flow: upload file -> create transcription -> poll -> fetch tokens -> delete both
the transcription and the uploaded file. Deletion runs on every exit path so no
audio or transcript is left on Soniox's servers.

Each returned token carries ``text``, ``start_ms``, ``end_ms``, ``confidence``
(0..1) and, with language identification enabled, ``language``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class SonioxError(RuntimeError):
    pass


class Soniox:
    def __init__(self, api_key: str, base_url: str = "https://api.soniox.com",
                 model: str = "stt-async-v5", session=None, poll_interval: float = 2.0,
                 poll_timeout: float = 1800.0, retries: int = 3):
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

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        delay = 2.0
        for attempt in range(self.retries + 1):
            response = self.session.request(method, url, **kwargs)
            if response.status_code in RETRY_STATUS and attempt < self.retries:
                log.warning("Soniox %s %s -> %s, retrying in %.0fs", method, path, response.status_code, delay)
                time.sleep(delay)
                delay *= 2
                # Re-open file uploads on retry: the stream has been consumed.
                if "files" in kwargs and hasattr(kwargs["files"].get("file", (None, None))[1], "seek"):
                    kwargs["files"]["file"][1].seek(0)
                continue
            if not response.ok:
                hint = ""
                if response.status_code in (401, 403) and ".eu." in self.base:
                    hint = " (the EU endpoint needs a key from an EU project)"
                raise SonioxError(f"Soniox {response.status_code}: {response.text[:400]}{hint}")
            return response
        raise SonioxError(f"Soniox: giving up on {method} {path}")

    def transcribe(self, audio: Path, language_hints: list[str], context_terms: list[str] | None = None,
                   context_text: str = "", reference: str = "voicememos-mcp") -> list[dict]:
        with open(audio, "rb") as fh:
            file_id = self._request("POST", "/v1/files", files={"file": (audio.name, fh)}, timeout=600).json()["id"]
        transcription_id = None
        try:
            body: dict = {
                "model": self.model,
                "file_id": file_id,
                "language_hints": language_hints,
                "enable_language_identification": True,
                "client_reference_id": reference,
            }
            context: dict = {}
            if context_terms:
                context["terms"] = context_terms
            if context_text:
                context["text"] = context_text
            if context:
                body["context"] = context
            transcription_id = self._request("POST", "/v1/transcriptions", json=body, timeout=60).json()["id"]

            started = time.monotonic()
            while True:
                status = self._request("GET", f"/v1/transcriptions/{transcription_id}", timeout=60).json()
                state = status.get("status")
                if state == "completed":
                    break
                if state == "error":
                    raise SonioxError(f"Soniox transcription failed: {status.get('error_message')}")
                if time.monotonic() - started > self.poll_timeout:
                    raise SonioxError("Soniox transcription timed out")
                time.sleep(self.poll_interval)

            transcript = self._request("GET", f"/v1/transcriptions/{transcription_id}/transcript", timeout=120)
            return transcript.json().get("tokens", [])
        finally:
            targets = ([f"/v1/transcriptions/{transcription_id}"] if transcription_id else []) + [f"/v1/files/{file_id}"]
            for path in targets:
                try:
                    response = self.session.delete(f"{self.base}{path}", timeout=60)
                    if not response.ok and response.status_code != 404:
                        log.warning("Soniox cleanup failed for %s: %s", path, response.status_code)
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
                    log.warning("Soniox cleanup failed for %s: %s", path, exc)
