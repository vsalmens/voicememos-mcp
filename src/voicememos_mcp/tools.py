"""Tool implementations, independent of the MCP framework (see ``server.py``).

The tools only read ``library.json``, the transcript files and job files, and
write transcription requests to the queue. They never open the Voice Memos
container, so the MCP host process does not need Full Disk Access.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import threading

from . import notation
from .config import Config
from .store import Store, render_markdown

MAX_LIMIT = 200
DEFAULT_MAX_CHARS = 20000


class ToolError(ValueError):
    pass


def _norm(value: str) -> str:
    return value.lower().replace("-", "").strip()


def _check_date(value: str | None, name: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ToolError(f"{name} must be YYYY-MM-DD") from exc


def _summary(memo: dict) -> dict:
    transcript = memo.get("transcript") or {}
    stats = transcript.get("stats") or {}
    return {
        "id": memo.get("short_id"),
        "title": memo.get("title"),
        "recorded": memo.get("recorded"),
        "duration_s": round(memo["duration_s"], 1) if memo.get("duration_s") else None,
        "status": memo.get("status"),
        "certainty": stats.get("certainty"),
        "marked_share": stats.get("marked_share"),
        "markers": len(transcript.get("markers") or []),
        "speakers": transcript.get("speakers") or None,
        "voiceprint": transcript.get("voiceprint") or None,
    }


class Tools:
    def __init__(self, cfg: Config, store: Store | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg.data_dir, cfg.output_dir)

    # ------------------------------------------------------------------ helpers
    def _memos(self, include_deleted: bool = False) -> list[dict]:
        memos = self.store.read_library().get("memos", [])
        return [m for m in memos if include_deleted or not m.get("deleted")]

    def _find(self, ref: str) -> dict:
        key = _norm(ref or "")
        if len(key) < 4:
            raise ToolError("memo id must have at least 4 characters (use the id from voicememos_list)")
        matches = []
        for memo in self._memos(include_deleted=True):
            keys = {_norm(memo.get("id", "")), _norm(memo.get("short_id", ""))}
            if memo.get("filename"):
                keys.add(_norm(os.path.splitext(memo["filename"])[0]))
            if any(k.startswith(key) for k in keys if k):
                matches.append(memo)
        if not matches:
            raise ToolError(f"no memo matches '{ref}'")
        if len(matches) > 1:
            raise ToolError(f"'{ref}' is ambiguous ({len(matches)} memos); use more characters")
        return matches[0]

    def _filter(self, memos: list[dict], since: str | None, until: str | None) -> list[dict]:
        since, until = _check_date(since, "since"), _check_date(until, "until")
        out = []
        for memo in memos:
            day = (memo.get("recorded") or "")[:10]
            if since and day < since:
                continue
            if until and day > until:
                continue
            out.append(memo)
        return out

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        library = self.store.read_library()
        memos = library.get("memos", [])
        counts: dict[str, int] = {}
        for memo in memos:
            counts[memo.get("status", "unknown")] = counts.get(memo.get("status", "unknown"), 0) + 1
        worker = self.store.read_worker()
        warnings = []
        if not worker:
            warnings.append("The worker has never run. Install the launchd job (see README) or run "
                            "'voicememos-mcp sync'.")
        elif not worker.get("library_readable", True):
            warnings.append(f"Worker cannot read Voice Memos: {worker.get('error')}")
        elif worker.get("running"):
            pass  # a run is in progress; library.json is refreshed every few memos
        else:
            last = worker.get("finished") or worker.get("started")
            try:
                age_h = (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(last)).total_seconds() / 3600
                if age_h > 2:
                    warnings.append(f"Last worker run was {age_h:.0f} h ago; the launchd job may not be running.")
            except (TypeError, ValueError):
                pass
        return {
            "library_updated": library.get("updated"),
            "memos": len(memos),
            "by_status": counts,
            "hidden_by_exclusion": library.get("excluded_count", 0),
            "worker": worker,
            "queued_requests": len(self.store.pending_requests()) if self.store.queue_dir.exists() else 0,
            "output_dir": str(self.cfg.output_dir),
            "engine": {"model": self.cfg.get("SONIOX_MODEL"), "languages": self.cfg.languages,
                       "threshold": self.cfg.threshold},
            "auto_transcribe": self.cfg.flag("VOICEMEMOS_AUTO_TRANSCRIBE"),
            "roam_enabled": self.cfg.flag("VOICEMEMOS_ROAM_ENABLED"),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ list
    def list_memos(self, limit: int = 20, offset: int = 0, since: str | None = None, until: str | None = None,
                   title: str | None = None, status: str | None = None) -> dict:
        memos = self._filter(self._memos(), since, until)
        if title:
            needle = title.casefold()
            memos = [m for m in memos if needle in (m.get("title") or "").casefold()]
        if status:
            memos = [m for m in memos if m.get("status") == status]
        limit = max(1, min(int(limit), MAX_LIMIT))
        offset = max(0, int(offset))
        page = memos[offset: offset + limit]
        return {"total": len(memos), "offset": offset, "memos": [_summary(m) for m in page],
                "next_offset": offset + limit if offset + limit < len(memos) else None}

    # ------------------------------------------------------------------ read
    def read(self, memo: str, format: str = "markdown", threshold: float | None = None,
             offset: int = 0, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
        entry = self._find(memo)
        info = _summary(entry)
        transcript = entry.get("transcript")
        if not transcript:
            return {**info, "text": None,
                    "note": "not transcribed yet; call voicememos_transcribe with this id"}
        data = self.store.load_transcript(transcript["json"])
        if data is None:
            raise ToolError("transcript file is missing; run voicememos_transcribe with redo=true")
        thr = float(threshold) if threshold is not None else self.cfg.threshold
        if format == "json":
            body = {k: v for k, v in data.items() if k != "source_tokens"}
            body["words"] = notation.annotate(data.get("words", []), thr)
            return {**info, "data": body}
        if format == "text":
            text = "\n\n".join(notation.render_blocks(data.get("words", []), thr, self.cfg.run_min,
                                                      self.cfg.pause_ms, data.get("speakers")))
        elif format == "markdown":
            text = render_markdown(data, thr, self.cfg.run_min, self.cfg.pause_ms)
        else:
            raise ToolError("format must be 'markdown', 'text' or 'json'")
        offset = max(0, int(offset))
        max_chars = max(1000, min(int(max_chars), 100000))
        chunk = text[offset: offset + max_chars]
        end = offset + len(chunk)
        return {**info, "threshold": thr, "file": transcript.get("markdown"), "text": chunk,
                "total_chars": len(text), "next_offset": end if end < len(text) else None}

    # ------------------------------------------------------------------ search
    def search(self, query: str, limit: int = 20, since: str | None = None, until: str | None = None,
               regex: bool = False, context_chars: int = 120) -> dict:
        if not query or not query.strip():
            raise ToolError("query is empty")
        if len(query) > 200:
            raise ToolError("query is too long (max 200 characters)")
        try:
            pattern = re.compile(query if regex else re.escape(query.strip()), re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        limit = max(1, min(int(limit), MAX_LIMIT))
        hits = []
        for memo in self._filter(self._memos(), since, until):
            title_hit = bool(memo.get("title") and pattern.search(memo["title"]))
            snippets = []
            transcript = memo.get("transcript")
            if transcript:
                data = self.store.load_transcript(transcript["json"]) or {}
                text = " ".join(notation.render_blocks(data.get("words", []), self.cfg.threshold,
                                                       self.cfg.run_min, self.cfg.pause_ms))
                for match in pattern.finditer(text):
                    start = max(0, match.start() - context_chars)
                    end = min(len(text), match.end() + context_chars)
                    snippets.append(("…" if start else "") + text[start:end] + ("…" if end < len(text) else ""))
                    if len(snippets) >= 3:
                        break
            if title_hit or snippets:
                hits.append({**_summary(memo), "title_match": title_hit, "snippets": snippets})
            if len(hits) >= limit:
                break
        return {"query": query, "hits": hits, "count": len(hits)}

    # ------------------------------------------------------------------ markers
    def markers(self, since: str | None = None, until: str | None = None, label: str | None = None,
                limit: int = 50) -> dict:
        out = []
        for memo in self._filter(self._memos(), since, until):
            for marker in (memo.get("transcript") or {}).get("markers") or []:
                if label and marker.get("label", "").casefold() != label.casefold():
                    continue
                out.append({"id": memo.get("short_id"), "recorded": memo.get("recorded"),
                            "title": memo.get("title"), **marker})
        limit = max(1, min(int(limit), MAX_LIMIT))
        return {"labels": sorted(set(self.cfg.markers.values())), "markers": out[:limit], "total": len(out)}

    # ------------------------------------------------------------------ transcribe
    def transcribe(self, memos: list[str] | None = None, since: str | None = None, until: str | None = None,
                   redo: bool = False, push_roam: bool = False) -> dict:
        memos = [m for m in (memos or []) if m and m.strip()]
        since, until = _check_date(since, "since"), _check_date(until, "until")
        if not memos and not since and not until:
            raise ToolError("give memo ids, or a date range (since/until)")
        resolved = [self._find(m)["id"] for m in memos]
        request = {"memos": resolved, "since": since, "until": until, "redo": bool(redo),
                   "push_roam": bool(push_roam)}
        self.store.ensure_dirs()
        job_id = self.store.enqueue(request)
        started_by = self._kick_worker()
        return {"job_id": job_id, "status": "queued", "worker": started_by,
                "hint": "check progress with voicememos_transcribe_status"}

    def second_read(self, memos: list[str] | None = None, since: str | None = None, until: str | None = None,
                    model: str | None = None, redo: bool = False, limit: int | None = None) -> dict:
        memos = [m for m in (memos or []) if m and m.strip()]
        since, until = _check_date(since, "since"), _check_date(until, "until")
        if not memos and not since and not until:
            raise ToolError("give memo ids, or a date range (since/until)")
        if model and not re.fullmatch(r"[A-Za-z0-9._-]{3,80}", model):
            raise ToolError("invalid model id")
        request = {"action": "second_read", "memos": [self._find(m)["id"] for m in memos], "since": since,
                   "until": until, "model": model, "redo": bool(redo),
                   "limit": max(1, int(limit)) if limit else None}
        self.store.ensure_dirs()
        job_id = self.store.enqueue(request)
        return {"job_id": job_id, "status": "queued", "worker": self._kick_worker(),
                "hint": "check progress with voicememos_transcribe_status, then voicememos_compare_readers"}

    def compare_readers(self, since: str | None = None, until: str | None = None, samples: int = 15) -> dict:
        from .report import build_report, write_report

        report = build_report(self.store, _check_date(since, "since"), _check_date(until, "until"),
                              max(0, min(int(samples), 50)))
        paths = write_report(self.store, report)
        summary = {lang: {k: v for k, v in group.items() if k != "samples"}
                   for lang, group in report["languages"].items()}
        return {"languages": summary, "report_file": paths["markdown"],
                "samples": {lang: group["samples"][:5] for lang, group in report["languages"].items()}}

    def voiceprint(self, action: str = "status", memos: list[str] | None = None, since: str | None = None,
                   until: str | None = None, name: str | None = None) -> dict:
        from .voiceprint import Voiceprint, suggest_enrollment

        if action == "status":
            summary = Voiceprint(self.store.voiceprint_path).summary()
            summary["configured_name"] = self.cfg.get("VOICEMEMOS_VOICEPRINT_NAME") or None
            summary["enrollment_suggestions"] = [_summary(m) for m in suggest_enrollment(self._memos())]
            summary["hint"] = ("enroll 4-6 memos where only you speak, from different years and places; "
                               "listen to them first if unsure")
            return summary
        memos = [m for m in (memos or []) if m and m.strip()]
        if action == "enroll":
            if len(memos) < 2:
                raise ToolError("give at least 2 memo ids (4-6 is better)")
            who = (name or self.cfg.get("VOICEMEMOS_VOICEPRINT_NAME") or "").strip()
            if not who or len(who) > 60:
                raise ToolError("give a name (or set VOICEMEMOS_VOICEPRINT_NAME)")
            ids = list(dict.fromkeys(self._find(m)["id"] for m in memos))
            if len(ids) < 2:
                raise ToolError("give at least 2 different memos")
            request = {"action": "enroll", "memos": ids, "name": who}
        elif action == "identify":
            since, until = _check_date(since, "since"), _check_date(until, "until")
            if not memos and not since and not until:
                raise ToolError("give memo ids, or a date range (since/until)")
            if not self.store.voiceprint_path.exists():
                raise ToolError("no voiceprint enrolled yet (action='enroll')")
            request = {"action": "identify", "memos": [self._find(m)["id"] for m in memos],
                       "since": since, "until": until}
        else:
            raise ToolError("action must be 'status', 'enroll' or 'identify'")
        self.store.ensure_dirs()
        job_id = self.store.enqueue(request)
        return {"job_id": job_id, "status": "queued", "worker": self._kick_worker(),
                "hint": "check progress with voicememos_transcribe_status"}

    def _kick_worker(self) -> str:
        mode = self.cfg.get("VOICEMEMOS_WORKER")
        if mode == "inprocess":
            from .worker import Worker

            threading.Thread(target=lambda: Worker(self.cfg, self.store).run(), daemon=True).start()
            return "started in this process"
        if mode == "launchd":
            label = self.cfg.get("VOICEMEMOS_LAUNCHD_LABEL")
            try:
                result = subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{label}"],
                                        capture_output=True, text=True, timeout=10, check=False)
                if result.returncode == 0:
                    return f"launchd job {label} started"
                return f"queued; launchd job {label} will pick it up (kickstart: {result.stderr.strip()[:120]})"
            except (OSError, subprocess.TimeoutExpired):
                return f"queued; launchd job {label} will pick it up"
        return "queued; run 'voicememos-mcp sync' to process"

    def transcribe_status(self, job_id: str | None = None) -> dict:
        if job_id:
            job = self.store.read_job(job_id.strip())
            if not job:
                raise ToolError(f"unknown job '{job_id}'")
            return job
        jobs = self.store.recent_jobs(10)
        return {"jobs": [{k: j.get(k) for k in ("job_id", "status", "created", "total", "done", "failed")}
                         for j in jobs]}
