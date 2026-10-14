"""Second reader: transcription with a Gemini model (Gemini API, ``x-goog-api-key``).

Gemini models return text without per-word confidence, so this reader is not used
for confidence marking. Its value is independence: where two engines disagree,
the transcript shows both readings (see ``align.py``).

Small files are sent inline; larger ones go through the Files API and are deleted
right after the request.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
from pathlib import Path

log = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com"
INLINE_LIMIT = 14 * 1024 * 1024  # request bodies are limited to about 20 MB after base64
RETRY_STATUS = {429, 500, 502, 503, 504}

LANGUAGE_NAMES = {"fi": "Finnish", "en": "English", "sv": "Swedish", "de": "German", "fr": "French",
                  "es": "Spanish", "it": "Italian", "et": "Estonian", "ru": "Russian"}

PROMPT = """Transcribe this audio recording verbatim.

Rules:
- Write exactly what is said, in the language it is spoken ({languages}). Never translate.
- Do not summarise, rephrase, correct grammar or remove filler words.
- Do not guess to fill gaps. If you are unsure of a word, write your best guess followed by " (?)".
  If a stretch is unintelligible, write [?].
- Plain text only: no timestamps, no speaker labels, no headings, no commentary.
- Use normal punctuation and start a new paragraph after a long pause.
{context}"""


class GeminiError(RuntimeError):
    pass


def build_prompt(languages: list[str], context_terms: list[str] | None = None, context_text: str = "") -> str:
    names = ", ".join(LANGUAGE_NAMES.get(code, code) for code in languages) or "the spoken language"
    context = []
    if context_text:
        context.append(f"Context about the recordings: {context_text}")
    if context_terms:
        context.append("Names and terms that may occur: " + ", ".join(context_terms))
    return PROMPT.format(languages=names, context="\n".join(context))


class Gemini:
    def __init__(self, api_key: str, model: str = "gemini-4-argon", session=None, retries: int = 4,
                 timeout: float = 600.0):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers["x-goog-api-key"] = api_key
        self.model = model
        self.retries = retries
        self.timeout = timeout

    def _post(self, url: str, **kwargs):
        delay = 5.0
        for attempt in range(self.retries + 1):
            response = self.session.post(url, **kwargs)
            if response.status_code in RETRY_STATUS and attempt < self.retries:
                log.warning("Gemini %s -> %s, retrying in %.0fs", url.split("?")[0], response.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if not response.ok:
                hint = " (billing or credits exhausted)" if response.status_code == 402 else ""
                raise GeminiError(f"Gemini {response.status_code}: {response.text[:400]}{hint}")
            return response
        raise GeminiError("Gemini: giving up")

    def _upload(self, audio: Path, mime: str) -> dict:
        size = audio.stat().st_size
        start = self._post(
            f"{API}/upload/v1beta/files",
            headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                     "X-Goog-Upload-Header-Content-Length": str(size),
                     "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"},
            json={"file": {"display_name": "voice-memo"}}, timeout=60,
        )
        upload_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise GeminiError("Gemini upload: no upload URL returned")
        with open(audio, "rb") as fh:
            done = self._post(upload_url, data=fh, timeout=self.timeout,
                              headers={"X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0",
                                       "Content-Length": str(size)})
        file = done.json().get("file", {})
        name = file.get("name")
        for _ in range(60):  # wait until processed
            if file.get("state") in (None, "ACTIVE"):
                return file
            if file.get("state") == "FAILED":
                raise GeminiError("Gemini upload: file processing failed")
            time.sleep(2)
            file = self.session.get(f"{API}/v1beta/{name}", timeout=60).json()
        raise GeminiError("Gemini upload: file did not become active")

    def _delete(self, name: str) -> None:
        try:
            response = self.session.delete(f"{API}/v1beta/{name}", timeout=60)
            if not response.ok and response.status_code != 404:
                log.warning("Gemini cleanup failed for %s: %s", name, response.status_code)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
            log.warning("Gemini cleanup failed for %s: %s", name, exc)

    def transcribe(self, audio: Path, prompt: str) -> dict:
        """Return ``{"text": ..., "usage": {...}, "model": ...}``."""
        mime = mimetypes.guess_type(audio.name)[0] or "audio/mp4"
        if audio.suffix.lower() == ".m4a":
            mime = "audio/mp4"
        uploaded = None
        try:
            if audio.stat().st_size <= INLINE_LIMIT:
                part = {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio.read_bytes()).decode()}}
            else:
                uploaded = self._upload(audio, mime)
                part = {"file_data": {"mime_type": uploaded.get("mimeType", mime), "file_uri": uploaded["uri"]}}
            body = {"contents": [{"role": "user", "parts": [part, {"text": prompt}]}],
                    "generationConfig": {"temperature": 0}}
            response = self._post(f"{API}/v1beta/models/{self.model}:generateContent", json=body,
                                  timeout=self.timeout).json()
        finally:
            if uploaded and uploaded.get("name"):
                self._delete(uploaded["name"])
        candidates = response.get("candidates") or []
        if not candidates:
            raise GeminiError(f"Gemini returned no candidates: {str(response.get('promptFeedback'))[:200]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        finish = candidates[0].get("finishReason")
        if finish and finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            log.warning("Gemini finished with %s; the transcript may be incomplete", finish)
        return {"text": text, "model": response.get("modelVersion", self.model),
                "usage": response.get("usageMetadata", {}), "finish_reason": finish}
