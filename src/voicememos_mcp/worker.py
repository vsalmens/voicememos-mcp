"""The worker: the only component that reads the Voice Memos container.

One run (``voicememos-mcp sync``):

1. read the library database and write a snapshot to ``library.json``,
2. process queued transcription requests from the MCP server,
3. transcribe new memos automatically (if enabled),
4. optionally append new transcripts to Roam,
5. record the result in ``worker.json``.

On macOS it runs from launchd, triggered by changes in the recordings folder or
the queue folder and every 30 minutes. Only this process needs Full Disk Access.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import align, notation, roam
from .audio import prepare_audio
from .config import Config
from .library import LibraryError, Memo, is_excluded, read_library
from .store import Store, now_iso

log = logging.getLogger(__name__)

Transcriber = Callable[[Path], list[dict]]
SecondReader = Callable[[Path], dict]


class Busy(RuntimeError):
    pass


@contextlib.contextmanager
def single_instance(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Busy("another worker run is in progress") from exc
        yield


def soniox_transcriber(cfg: Config) -> Transcriber:
    from .soniox import Soniox

    key = cfg.secret("SONIOX_API_KEY")
    if not key:
        raise RuntimeError("SONIOX_API_KEY is not set (environment or Keychain)")
    client = Soniox(key, cfg.get("SONIOX_BASE_URL"), cfg.get("SONIOX_MODEL"))

    def transcribe(audio: Path) -> list[dict]:
        return client.transcribe(audio, cfg.languages, cfg.context_terms, cfg.context_text,
                                 diarization=cfg.flag("VOICEMEMOS_DIARIZATION"))

    return transcribe


def gemini_reader(cfg: Config, model: str | None = None) -> SecondReader:
    from .gemini import Gemini, build_prompt

    key = cfg.secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (environment or Keychain)")
    client = Gemini(key, model or cfg.get("VOICEMEMOS_SECOND_READER_MODEL"))
    prompt = build_prompt(cfg.languages, cfg.context_terms, cfg.context_text)

    def read(audio: Path) -> dict:
        return client.transcribe(audio, prompt)

    return read


def _vp_short(summary: dict | None) -> dict | None:
    if not summary:
        return None
    short = {"result": summary.get("result")}
    if summary.get("score") is not None:
        short["score"] = summary["score"]
    if summary.get("speaker") is not None:
        short["speaker"] = summary["speaker"]
    return short


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def _in_range(memo: Memo, since: str | None, until: str | None) -> bool:
    day = (memo.recorded or "")[:10]
    if since and (not day or day < since):
        return False
    if until and (not day or day > until):
        return False
    return True


class Worker:
    def __init__(self, cfg: Config, store: Store | None = None, transcriber: Transcriber | None = None,
                 roam_session=None, clock: Callable[[], float] = time.time,
                 second_reader: SecondReader | None = None, embedder=None, audio_loader=None):
        self.cfg = cfg
        self.store = store or Store(cfg.data_dir, cfg.output_dir)
        self._transcriber = transcriber
        self._second_reader = second_reader
        self._embedder = embedder
        self._audio_loader = audio_loader
        self.roam_session = roam_session
        self.clock = clock
        self.state: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = soniox_transcriber(self.cfg)
        return self._transcriber

    def second_reader(self, model: str | None = None, on_demand: bool = False) -> SecondReader | None:
        """The reader used after each new transcription (``VOICEMEMOS_SECOND_READER=gemini``), or,
        with ``on_demand``, for explicit second-read requests even when automatic reading is off."""
        if self._second_reader is not None and model is None:
            return self._second_reader
        if model or on_demand or self.cfg.get("VOICEMEMOS_SECOND_READER") == "gemini":
            reader = gemini_reader(self.cfg, model)
            if model is None:
                self._second_reader = reader
            return reader
        return None

    def _apply_second_reader(self, reader: SecondReader, audio: Path, memo: Memo, words: list[dict]) -> tuple[list[dict], dict]:
        """Run the second reader and align it. Returns (words, reader_record). Never raises."""
        record: dict = {"role": "second", "provider": "gemini", "created": now_iso()}
        max_s = float(self.cfg.get("VOICEMEMOS_SECOND_READER_MAX_MIN")) * 60
        if memo.duration_s and memo.duration_s > max_s:
            record["error"] = f"skipped: longer than {max_s / 60:.0f} min"
            return words, record
        try:
            result = reader(audio)
            aligned, metrics = align.align(words, result["text"], self.cfg.threshold)
            record.update(model=result.get("model"), text=result["text"], usage=result.get("usage", {}),
                          finish_reason=result.get("finish_reason"), agreement=metrics)
            return aligned, record
        except Exception as exc:  # noqa: BLE001 - the primary transcript must survive
            log.exception("Second reader failed for %s", memo.id)
            record["error"] = str(exc)[:300]
            return words, record

    # ------------------------------------------------------------------ voiceprint
    def embedder(self):
        if isinstance(self._embedder, Exception):  # failed earlier in this run: do not retry per memo
            raise RuntimeError(f"voiceprint model unavailable: {self._embedder}")
        if self._embedder is None:
            from .voiceprint import SherpaEmbedder, ensure_model

            try:
                model = self.cfg.get("VOICEMEMOS_VOICEPRINT_MODEL")
                self._embedder = SherpaEmbedder(ensure_model(self.store.models_dir, model))
            except Exception as exc:
                self._embedder = exc
                raise
        return self._embedder

    def load_audio(self, path: Path):
        if self._audio_loader is not None:
            return self._audio_loader(path)
        from .voiceprint import load_audio

        return load_audio(path)

    def _identify(self, audio: Path, words: list[dict], speakers: dict) -> tuple[dict, dict | None]:
        """Apply the voiceprint if one is enrolled. Never raises."""
        from .voiceprint import Voiceprint, identify

        voiceprint = Voiceprint(self.store.voiceprint_path).load()
        if not voiceprint:
            return speakers, None
        try:
            return identify(self.load_audio(audio), words, speakers, voiceprint, self.embedder())
        except Exception as exc:  # noqa: BLE001 - the transcript must survive
            log.exception("Voiceprint identification failed")
            return speakers, {"name": voiceprint.get("name"), "result": "error", "error": str(exc)[:200]}

    def enroll_voiceprint(self, memos: list[Memo], name: str) -> dict:
        from .voiceprint import Voiceprint, enroll

        if not name:
            raise RuntimeError("no name given (set VOICEMEMOS_VOICEPRINT_NAME or pass a name)")
        samples, words = {}, {}
        with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
            for memo in memos:
                if not memo.local or not memo.filename:
                    raise RuntimeError(f"memo {memo.short_id} is not downloaded")
                samples[memo.id] = self.load_audio(prepare_audio(self.cfg.recordings_dir / memo.filename, Path(tmp)))
                entry = self.state.get(memo.id, {})
                data = self.store.load_transcript(entry["json"]) if entry.get("ok") else None
                if data and data.get("speakers"):
                    raise RuntimeError(f"memo {memo.short_id} has {len(data['speakers'])} speakers; "
                                       "enroll only memos where you speak alone")
                if data:
                    words[memo.id] = data.get("words", [])
            # The readers decode lazily from the temporary files, so embed before they are removed.
            result = enroll(name, self.cfg.get("VOICEMEMOS_VOICEPRINT_MODEL"), samples, self.embedder(), words)
        Voiceprint(self.store.voiceprint_path).save(result)
        return Voiceprint(self.store.voiceprint_path).summary()

    def identify_memo(self, memo: Memo) -> dict:
        """Apply the voiceprint to an existing transcript (no new transcription)."""
        item = {"id": memo.id, "short_id": memo.short_id, "title": memo.title, "recorded": memo.recorded}
        entry = self.state.get(memo.id, {})
        if not entry.get("ok"):
            return {**item, "status": "not_transcribed"}
        if not memo.local or not memo.filename:
            return {**item, "status": "not_downloaded"}
        try:
            data = self.store.load_transcript(entry["json"])
            if data is None:
                return {**item, "status": "failed", "error": "transcript file missing"}
            words = data.get("words", [])
            with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
                audio = prepare_audio(self.cfg.recordings_dir / memo.filename, Path(tmp))
                speakers, summary = self._identify(audio, words, notation.speaker_labels(words))
            if summary is None:
                return {**item, "status": "failed", "error": "no voiceprint enrolled"}
            if summary.get("result") == "error":
                return {**item, "status": "failed", "error": summary.get("error")}
            data["speakers"], data["voiceprint"] = speakers, summary
            result = self.store.rewrite_transcript(entry["json"], data, self.cfg.threshold, self.cfg.run_min,
                                                   self.cfg.pause_ms, self.cfg.markers)
        except Exception as exc:  # noqa: BLE001 - a failing memo must not block the queue
            log.exception("Voiceprint identification failed for %s", memo.id)
            return {**item, "status": "failed", "error": str(exc)[:300]}
        entry.update(stats=result["stats"], markers=result["markers"], speakers=len(speakers),
                     voiceprint=_vp_short(summary))
        self.state[memo.id] = entry
        return {**item, "status": "identified", "voiceprint": _vp_short(summary)}

    def _excluded(self, memo: Memo) -> bool:
        return is_excluded(memo, self.cfg.exclude_title_regex, self.cfg.exclude_folders)

    def _status(self, memo: Memo) -> str:
        entry = self.state.get(memo.id, {})
        if memo.deleted:
            return "deleted"
        if entry.get("ok"):
            return "transcribed"
        if not memo.local:
            return "not_downloaded"
        if entry.get("attempts", 0) >= self.cfg.max_attempts:
            return "failed"
        if memo.duration_s and memo.duration_s > self.cfg.max_duration_s:
            return "too_long"
        if self.cfg.since and not _in_range(memo, self.cfg.since, None):
            return "not_transcribed"
        return "pending" if self.cfg.flag("VOICEMEMOS_AUTO_TRANSCRIBE") else "not_transcribed"

    def _snapshot(self, memos: list[Memo]) -> dict:
        items, excluded = [], 0
        for memo in memos:
            if self._excluded(memo):
                excluded += 1  # excluded memos are not listed at all, not even their titles
                continue
            entry = self.state.get(memo.id, {})
            item = memo.to_dict()
            item["status"] = self._status(memo)
            if entry.get("ok"):
                item["transcript"] = {k: entry.get(k) for k in ("json", "markdown", "stats", "markers", "at",
                                                                "language", "second_reader", "speakers",
                                                                "voiceprint")}
            elif entry.get("error"):
                item["error"] = entry["error"]
            items.append(item)
        return {"updated": now_iso(), "recordings_dir": str(self.cfg.recordings_dir),
                "excluded_count": excluded, "memos": items}

    # ------------------------------------------------------------------ one memo
    def transcribe_memo(self, memo: Memo, force: bool = False, push_roam: bool = False,
                        dry_run: bool = False) -> dict:
        """Transcribe one memo. Returns an item dict with ``status`` and, on success, output paths."""
        item = {"id": memo.id, "short_id": memo.short_id, "title": memo.title, "recorded": memo.recorded}
        if self._excluded(memo):
            return {**item, "status": "excluded", "title": None}
        if memo.deleted:
            return {**item, "status": "deleted"}
        if not memo.local or not memo.filename:
            return {**item, "status": "not_downloaded",
                    "error": "audio is still in iCloud; play the memo once in Voice Memos on this Mac"}
        path = self.cfg.recordings_dir / memo.filename
        fingerprint = _fingerprint(path)
        entry = self.state.get(memo.id, {})
        if not force and entry.get("ok") and entry.get("fp") == fingerprint:
            return {**item, "status": "already_transcribed", "markdown": entry.get("markdown")}

        try:
            second_record = None
            with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
                audio = prepare_audio(path, Path(tmp))
                tokens = self.transcriber(audio)
                words, had_confidence = notation.tokens_to_words(tokens)
                reader = None if dry_run else self.second_reader()
                if reader is not None:
                    words, second_record = self._apply_second_reader(reader, audio, memo, words)
                speakers = notation.speaker_labels(words)
                voice = None
                if not dry_run:
                    speakers, voice = self._identify(audio, words, speakers)
            if dry_run:
                text = "\n\n".join(notation.render_blocks(words, self.cfg.threshold, self.cfg.run_min,
                                                          self.cfg.pause_ms, speakers))
                return {**item, "status": "dry_run", "stats": notation.stats(words, self.cfg.threshold),
                        "text": text, "confidence_available": had_confidence}
            engine = {"provider": "soniox", "model": self.cfg.get("SONIOX_MODEL"),
                      "language_hints": self.cfg.languages}
            extra = {"language": align.dominant_language(words), "speakers": speakers, "voiceprint": voice,
                     "readers": [{"role": "primary", **engine, "created": now_iso()}]
                     + ([second_record] if second_record else [])}
            result = self.store.write_transcript(
                memo.to_dict(), words, engine, had_confidence, self.cfg.threshold,
                self.cfg.run_min, self.cfg.pause_ms, self.cfg.markers, source_tokens=tokens, extra=extra,
            )
            result["language"] = extra["language"]
            result["speakers"] = len(speakers)
            if voice:
                result["voiceprint"] = _vp_short(voice)
            if second_record and not second_record.get("error"):
                result["second_reader"] = {"model": second_record.get("model"),
                                           "agreement": second_record["agreement"]["agreement"]}
            new_entry = {"fp": fingerprint, "ok": True, "at": now_iso(), **result}
            if entry.get("roam"):
                new_entry["roam"] = entry["roam"]
            self.state[memo.id] = new_entry
            item.update(status="transcribed", markdown=result["markdown"], stats=result["stats"])
            if push_roam and self.cfg.flag("VOICEMEMOS_ROAM_ENABLED") and not new_entry.get("roam"):
                self._push_roam(memo, words, result, new_entry, item)
            return item
        except Exception as exc:  # noqa: BLE001 - one failing memo must not stop the run
            log.exception("Transcription failed for %s", memo.id)
            if not dry_run:
                self.state[memo.id] = {**entry, "ok": False, "fp": fingerprint,
                                       "attempts": entry.get("attempts", 0) + 1, "error": str(exc)[:300],
                                       "at": now_iso()}
            return {**item, "status": "failed", "error": str(exc)[:300]}

    def second_read_memo(self, memo: Memo, model: str | None = None, redo: bool = False) -> dict:
        """Add a second reader to an existing transcript without calling the primary engine again."""
        item = {"id": memo.id, "short_id": memo.short_id, "title": memo.title, "recorded": memo.recorded}
        entry = self.state.get(memo.id, {})
        if self._excluded(memo) or memo.deleted:
            return {**item, "status": "skipped"}
        if not entry.get("ok"):
            return {**item, "status": "not_transcribed", "error": "transcribe it first"}
        data = self.store.load_transcript(entry["json"])
        if data is None:
            return {**item, "status": "failed", "error": "transcript file missing"}
        if not redo and any(r.get("role") == "second" and not r.get("error") for r in data.get("readers", [])):
            return {**item, "status": "already_read", "markdown": entry.get("markdown")}
        if not memo.local or not memo.filename:
            return {**item, "status": "not_downloaded"}
        try:
            reader = self.second_reader(model, on_demand=True)
            if reader is None:
                return {**item, "status": "failed", "error": "no second reader configured"}
            with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
                audio = prepare_audio(self.cfg.recordings_dir / memo.filename, Path(tmp))
                words, record = self._apply_second_reader(reader, audio, memo, data.get("words", []))
        except Exception as exc:  # noqa: BLE001
            return {**item, "status": "failed", "error": str(exc)[:300]}
        if record.get("error"):
            return {**item, "status": "failed", "error": record["error"]}
        readers = [r for r in data.get("readers", []) if r.get("role") != "second"] + [record]
        if not any(r.get("role") == "primary" for r in readers):
            readers.insert(0, {"role": "primary", **data.get("engine", {})})
        data["readers"] = readers
        data["words"] = words
        data.setdefault("language", align.dominant_language(words))
        result = self.store.rewrite_transcript(entry["json"], data, self.cfg.threshold, self.cfg.run_min,
                                               self.cfg.pause_ms, self.cfg.markers)
        entry.update(stats=result["stats"], markers=result["markers"], language=data["language"],
                     speakers=len(data.get("speakers") or {}))
        entry["second_reader"] = {"model": record.get("model"), "agreement": record["agreement"]["agreement"],
                                  "at": record["created"]}
        self.state[memo.id] = entry
        return {**item, "status": "second_read", "agreement": record["agreement"], "markdown": result["markdown"]}

    def _push_roam(self, memo: Memo, words: list[dict], result: dict, entry: dict, item: dict) -> None:
        graph, token = self.cfg.get("ROAM_GRAPH"), self.cfg.secret("ROAM_API_TOKEN")
        if not (graph and token and memo.recorded):
            item["roam_error"] = "ROAM_GRAPH or ROAM_API_TOKEN missing"
            return
        try:
            blocks = notation.render_blocks(words, self.cfg.threshold, self.cfg.run_min, self.cfg.pause_ms,
                                            notation.speaker_labels(words))
            parent = roam.heading(self.cfg.get("VOICEMEMOS_ROAM_HEADING"), memo.to_dict(),
                                  notation.stats_line(result["stats"]))
            roam.append_daily_note(graph, token, dt.date.fromisoformat(memo.recorded[:10]), parent,
                                   blocks or ["(no speech recognised)"], session=self.roam_session)
            entry["roam"] = now_iso()
            item["roam"] = True
        except Exception as exc:  # noqa: BLE001 - the transcript is already saved
            entry["roam_error"] = str(exc)[:300]
            item["roam_error"] = str(exc)[:300]

    # ------------------------------------------------------------------ queue
    def _resolve_request(self, request: dict, memos: list[Memo]) -> list[Memo]:
        wanted = [str(x).lower().replace("-", "") for x in request.get("memos") or []]
        since, until = request.get("since"), request.get("until")
        selected = []
        for memo in memos:
            if self._excluded(memo) or memo.deleted:
                continue
            if wanted:
                keys = {memo.id.lower().replace("-", ""), (Path(memo.filename).stem.lower() if memo.filename else "")}
                if not any(w and any(k.startswith(w) for k in keys if k) for w in wanted):
                    continue
            elif not (since or until or request.get("all")):
                continue
            if not _in_range(memo, since, until):
                continue
            selected.append(memo)
        selected.sort(key=lambda m: m.recorded or "")
        return selected

    def _run_enroll_job(self, job: dict, request: dict, memos: list[Memo]) -> None:
        job.update(status="running", started=now_iso())
        self.store.write_job(job)
        targets = self._resolve_request({"memos": request.get("memos") or []}, memos)
        try:
            if len(targets) != len(request.get("memos") or []):
                raise RuntimeError("some memos were not found (excluded, deleted or unknown)")
            job["voiceprint"] = self.enroll_voiceprint(targets, request.get("name")
                                                       or self.cfg.get("VOICEMEMOS_VOICEPRINT_NAME"))
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("Voiceprint enrollment failed")
            job.update(status="failed", error=str(exc)[:300])
        job["finished"] = now_iso()
        self.store.write_job(job)
        self.store.finish_request(job["job_id"])

    def process_queue(self, memos: list[Memo]) -> list[dict]:
        jobs = []
        for request_doc in self.store.pending_requests():
            job = self.store.read_job(request_doc["job_id"]) or request_doc
            request = job.get("request", {})
            action = request.get("action")
            if action == "enroll":
                self._run_enroll_job(job, request, memos)
                jobs.append(job)
                continue
            targets = self._resolve_request(request, memos)
            if action == "identify":
                targets = [m for m in targets if self.state.get(m.id, {}).get("ok")]
            if request.get("action") == "second_read":
                targets = [m for m in targets if self.state.get(m.id, {}).get("ok")]
                if not request.get("redo"):
                    targets = [m for m in targets if not self.state[m.id].get("second_reader")]
            if request.get("limit"):
                targets = targets[: int(request["limit"])]
            job.update(status="running", started=now_iso(), total=len(targets), done=0, failed=0, items=[])
            self.store.write_job(job)
            push = bool(request.get("push_roam")) or self.cfg.flag("VOICEMEMOS_ROAM_BACKFILL")
            second = request.get("action") == "second_read"
            for memo in targets:
                if action == "identify":
                    item = self.identify_memo(memo)
                elif second:
                    item = self.second_read_memo(memo, request.get("model"), bool(request.get("redo")))
                else:
                    item = self.transcribe_memo(memo, force=bool(request.get("redo")), push_roam=push)
                job["items"].append({k: item.get(k) for k in ("short_id", "title", "recorded", "status",
                                                             "error", "markdown")})
                job["done"] += 1
                job["failed"] += item["status"] == "failed"
                if second and item.get("agreement"):
                    job["items"][-1]["agreement"] = item["agreement"].get("agreement")
                if item.get("voiceprint"):
                    job["items"][-1]["voiceprint"] = item["voiceprint"]
                self.store.write_job(job)
                self.store.write_state(self.state)
            job.update(status="done", finished=now_iso())
            if not targets:
                job["note"] = "no memos matched the request"
            self.store.write_job(job)
            self.store.finish_request(job["job_id"])
            jobs.append(job)
        return jobs

    # ------------------------------------------------------------------ automatic
    def auto_candidates(self, memos: list[Memo]) -> list[Memo]:
        if not self.cfg.flag("VOICEMEMOS_AUTO_TRANSCRIBE"):
            return []
        now = self.clock()
        out = []
        for memo in memos:
            if self._status(memo) != "pending":
                continue
            path = self.cfg.recordings_dir / memo.filename
            try:
                if now - path.stat().st_mtime < self.cfg.min_age_s:
                    continue  # iCloud may still be writing the file
                entry = self.state.get(memo.id, {})
                if entry.get("fp") == _fingerprint(path) and entry.get("attempts", 0) >= self.cfg.max_attempts:
                    continue
            except FileNotFoundError:
                continue
            out.append(memo)
        out.sort(key=lambda m: m.recorded or "")
        return out

    # ------------------------------------------------------------------ second reader batch
    def run_second_read(self, only: list[str] | None = None, since: str | None = None, until: str | None = None,
                        limit: int | None = None, model: str | None = None, redo: bool = False) -> dict:
        """Add the second reader to existing transcripts (synchronously, under the worker lock)."""
        self.store.ensure_dirs()
        try:
            with single_instance(self.store.data_dir / "worker.lock"):
                self.state = self.store.read_state()
                memos = read_library(self.cfg.recordings_dir)
                request = {"memos": only or [], "since": since, "until": until, "all": not (only or since or until)}
                targets = [m for m in self._resolve_request(request, memos) if self.state.get(m.id, {}).get("ok")]
                items = []
                for memo in targets:
                    if limit and sum(1 for i in items if i["status"] == "second_read") >= limit:
                        break
                    items.append(self.second_read_memo(memo, model, redo))
                    self.store.write_state(self.state)
                self.store.write_library(self._snapshot(memos))
                return {"ok": True, "items": items}
        except Busy as exc:
            return {"ok": False, "error": str(exc)}
        except LibraryError as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ run
    def run(self, dry_run: bool = False, limit: int | None = None, only: list[str] | None = None,
            redo: bool = False) -> dict:
        self.store.ensure_dirs()
        started = now_iso()
        try:
            with single_instance(self.store.data_dir / "worker.lock"):
                return self._run(started, dry_run, limit, only, redo)
        except Busy as exc:
            return {"ok": True, "skipped": str(exc)}

    def _run(self, started: str, dry_run: bool, limit: int | None, only: list[str] | None, redo: bool) -> dict:
        self.state = self.store.read_state()
        try:
            memos = read_library(self.cfg.recordings_dir)
        except LibraryError as exc:
            summary = {"ok": False, "started": started, "finished": now_iso(), "error": str(exc),
                       "library_readable": False}
            if not dry_run:
                self.store.write_worker(summary)
            return summary

        summary: dict = {"ok": True, "started": started, "library_readable": True, "memos": len(memos),
                         "jobs": [], "transcribed": [], "failed": []}
        if not dry_run:
            self.store.write_library(self._snapshot(memos))
            self.store.write_worker({"ok": True, "running": True, "started": started, "library_readable": True,
                                     "memos": len(memos)})

        if only:
            targets = self._resolve_request({"memos": only}, memos)
            for memo in targets[: limit or None]:
                item = self.transcribe_memo(memo, force=redo, dry_run=dry_run)
                summary["transcribed" if item["status"] in ("transcribed", "dry_run") else "failed"].append(item)
        else:
            if not dry_run:
                summary["jobs"] = [{"job_id": j["job_id"], "total": j.get("total"), "failed": j.get("failed")}
                                   for j in self.process_queue(memos)]
            count = 0
            for memo in self.auto_candidates(memos):
                if limit and count >= limit:
                    break
                item = self.transcribe_memo(memo, push_roam=True, dry_run=dry_run)
                bucket = "transcribed" if item["status"] in ("transcribed", "dry_run") else "failed"
                summary[bucket].append(item)
                count += 1
                if not dry_run:
                    self.store.write_state(self.state)
                    if count % 5 == 0:  # let the MCP tools see progress during long runs
                        self.store.write_library(self._snapshot(memos))

        if not dry_run:
            self.store.write_state(self.state)
            self.store.write_library(self._snapshot(memos))
            summary["finished"] = now_iso()
            self.store.write_worker({k: v for k, v in summary.items() if k not in ("transcribed", "failed")}
                                    | {"transcribed_count": len(summary["transcribed"]),
                                       "failed_count": len(summary["failed"])})
        return summary
