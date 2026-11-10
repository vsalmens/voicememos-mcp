"""On-disk layout.

``OUTPUT_DIR`` (readable by people and by the MCP server)::

    2026/2026-10-13_2135_some-title_1a2b3c4d.md    rendered transcript
    2026/2026-10-13_2135_some-title_1a2b3c4d.json  canonical word data

``DATA_DIR`` (internal state)::

    library.json   snapshot of the Voice Memos library, written by the worker
    state.json     per-memo processing state (fingerprint, attempts, outputs)
    worker.json    last worker run: time, result, permission status
    queue/         transcription requests written by the MCP server
    jobs/          job progress written by the worker (also one job per part of a memo in parts)
    tmp/<memo id>/ parts of a memo longer than the engine accepts, and their raw results;
                   removed when the memo is done, kept after a failure for diagnosis
    logs/          worker logs (launchd)

The split lets the MCP server run without access to the Voice Memos container:
only the worker reads it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
import unicodedata
import uuid
from pathlib import Path

from . import notation
from .segment import clock

SCHEMA = "voicememos-mcp/transcript@1"
JOB_ID = re.compile(r"[0-9a-f]{12}")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def slugify(text: str | None, limit: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()[:limit].rstrip("-")
    return slug or "memo"


class Store:
    def __init__(self, data_dir: Path, output_dir: Path, folder_dirs: dict[str, Path] | None = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # Transcripts of some folders live apart (a Plaud recorder's, say): folder name -> directory.
        # Their paths are kept absolute in the state; paths under output_dir stay relative as before.
        self.folder_dirs = {name.casefold(): path for name, path in (folder_dirs or {}).items()}
        self.stamp_every_s = 0  # paragraph time stamps in the Markdown files (set from config)

    # ---- paths
    @property
    def library_path(self) -> Path:
        return self.data_dir / "library.json"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def worker_path(self) -> Path:
        return self.data_dir / "worker.json"

    @property
    def voiceprint_path(self) -> Path:
        return self.data_dir / "voiceprint.json"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "queue"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.queue_dir, self.jobs_dir, self.data_dir / "logs", self.output_dir,
                     *self.folder_dirs.values()):
            path.mkdir(parents=True, exist_ok=True)

    # ---- library / state / worker
    def read_library(self) -> dict:
        return read_json(self.library_path, {"memos": [], "updated": None})

    def write_library(self, data: dict) -> None:
        write_json_atomic(self.library_path, data)

    def read_state(self) -> dict:
        return read_json(self.state_path, {})

    def write_state(self, state: dict) -> None:
        write_json_atomic(self.state_path, state)

    def read_worker(self) -> dict:
        return read_json(self.worker_path, {})

    def write_worker(self, data: dict) -> None:
        write_json_atomic(self.worker_path, data)

    # ---- transcripts
    def root_for(self, memo: dict) -> Path:
        return self.folder_dirs.get(str(memo.get("folder") or "").casefold(), self.output_dir)

    def _stored(self, path: Path) -> str:
        """A transcript path as the state keeps it: relative under output_dir, absolute elsewhere."""
        return str(path.relative_to(self.output_dir)) if self.output_dir in path.parents else str(path)

    def _resolve(self, stored: str) -> Path | None:
        """The file behind a stored path, or None when it is outside every transcript directory."""
        path = (self.output_dir / stored).resolve()  # an absolute stored path replaces output_dir
        roots = [self.output_dir.resolve(), *(d.resolve() for d in self.folder_dirs.values())]
        return path if any(root in path.parents for root in roots) else None

    def relocate(self, memo: dict, entry: dict) -> bool:
        """Move a memo's transcript pair to the directory its folder now belongs to (the setting
        changed, or the transcript was written before the folder had a directory of its own).
        Updates the paths in ``entry``; True when something moved."""
        current = self._resolve(entry.get("json") or "")
        root = self.root_for(memo).resolve()
        if current is None or root in current.parents or not current.exists():
            return False
        target = self.root_for(memo) / current.parent.name / current.name
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", ".md"):
            if current.with_suffix(suffix).exists():
                os.replace(current.with_suffix(suffix), target.with_suffix(suffix))
        entry["json"], entry["markdown"] = self._stored(target), self._stored(target.with_suffix(".md"))
        return True

    def transcript_base(self, memo: dict) -> Path:
        recorded = memo.get("recorded")
        when = dt.datetime.fromisoformat(recorded) if recorded else dt.datetime.now().astimezone()
        short = memo["id"].replace("-", "")[:8].lower()
        name = f"{when:%Y-%m-%d_%H%M}_{slugify(memo.get('title'))}_{short}"
        return self.root_for(memo) / f"{when:%Y}" / name

    def write_transcript(self, memo: dict, words: list[dict], engine: dict, had_confidence: bool,
                         threshold: float, run_min: int, pause_ms: int, markers,
                         source_tokens: list[dict] | None = None, extra: dict | None = None) -> dict:
        """Write the JSON (canonical) and Markdown (rendered) files. Returns relative paths.

        ``source_tokens`` is the engine's raw response, kept verbatim: it is the only
        lossless record, and every other format can be regenerated from it.
        """
        base = self.transcript_base(memo)
        base.parent.mkdir(parents=True, exist_ok=True)
        speakers = (extra or {}).get("speakers")
        text = "\n\n".join(notation.render_blocks(words, threshold, run_min, pause_ms, speakers))
        data = {
            "schema": SCHEMA,
            "memo": {k: memo.get(k) for k in ("id", "title", "recorded", "duration_s", "filename", "folder")},
            "engine": engine,
            "confidence_available": had_confidence,
            "created": now_iso(),
            "stats": notation.stats(words, threshold),
            "markers": notation.extract_markers(text, markers, words, threshold),
            "words": notation.annotate(words, threshold),
        }
        if extra:
            data.update(extra)
        if source_tokens is not None:
            data["source_tokens"] = source_tokens
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        # Remove outputs of an earlier run under a different name (title changed).
        for old in base.parent.glob(f"*_{base.name.rsplit('_', 1)[-1]}.*"):
            if old.with_suffix("") != base and old.suffix in (".md", ".json"):
                old.unlink(missing_ok=True)
        write_json_atomic(json_path, data)
        md_path.write_text(render_markdown(data, threshold, run_min, pause_ms, self.stamp_every_s), encoding="utf-8")
        return {
            "json": self._stored(json_path),
            "markdown": self._stored(md_path),
            "stats": data["stats"],
            "markers": data["markers"],
        }

    def rewrite_transcript(self, relative_json: str, data: dict, threshold: float, run_min: int,
                           pause_ms: int, markers) -> dict:
        """Re-derive stats, markers and flags from ``data["words"]`` and rewrite both files in place."""
        json_path = self._resolve(relative_json)
        if json_path is None:
            raise ValueError("transcript path outside the output directory")
        words = data.get("words", [])
        text = "\n\n".join(notation.render_blocks(words, threshold, run_min, pause_ms, data.get("speakers")))
        data["stats"] = notation.stats(words, threshold)
        data["markers"] = notation.extract_markers(text, markers, words, threshold)
        data["words"] = notation.annotate(words, threshold)
        data["updated"] = now_iso()
        write_json_atomic(json_path, data)
        json_path.with_suffix(".md").write_text(render_markdown(data, threshold, run_min, pause_ms, self.stamp_every_s),
                                                encoding="utf-8")
        return {"json": relative_json, "markdown": str(Path(relative_json).with_suffix(".md")),
                "stats": data["stats"], "markers": data["markers"]}

    def load_transcript(self, relative_json: str) -> dict | None:
        path = self._resolve(relative_json)
        return read_json(path) if path is not None else None

    # ---- queue / jobs
    def enqueue(self, request: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {"job_id": job_id, "status": "queued", "created": now_iso(), "request": request}
        write_json_atomic(self.jobs_dir / f"{job_id}.json", job)
        write_json_atomic(self.queue_dir / f"{job_id}.json", job)
        return job_id

    def pending_requests(self) -> list[dict]:
        items = []
        for path in sorted(self.queue_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            data = read_json(path)
            if data and JOB_ID.fullmatch(str(data.get("job_id", ""))) and data["job_id"] == path.stem:
                items.append(data)
        return items

    def finish_request(self, job_id: str) -> None:
        if JOB_ID.fullmatch(job_id or ""):
            (self.queue_dir / f"{job_id}.json").unlink(missing_ok=True)

    def read_job(self, job_id: str) -> dict | None:
        if not JOB_ID.fullmatch(job_id or ""):
            return None
        return read_json(self.jobs_dir / f"{job_id}.json")

    def write_job(self, job: dict) -> None:
        if not JOB_ID.fullmatch(job.get("job_id", "")):
            raise ValueError("invalid job id")
        write_json_atomic(self.jobs_dir / f"{job['job_id']}.json", job)

    def recent_jobs(self, limit: int = 10) -> list[dict]:
        paths = sorted(self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [j for j in (read_json(p) for p in paths[:limit]) if j]


def render_markdown(data: dict, threshold: float, run_min: int = 4, pause_ms: int = 2500,
                    stamp_every_s: int = 0) -> str:
    memo = data["memo"]
    words = data.get("words", [])
    s = notation.stats(words, threshold)
    blocks = notation.render_blocks(words, threshold, run_min, pause_ms, data.get("speakers"), stamp_every_s)
    text = "\n\n".join(blocks)
    markers = data.get("markers", [])
    title = memo.get("title") or "Voice memo"
    recorded = memo.get("recorded") or ""
    when = recorded.replace("T", " ")[:16]
    duration = memo.get("duration_s") or 0
    segments = data.get("segments") or {}
    front = [
        "---",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"recorded: {recorded}",
        f"duration_s: {int(duration)}",
        f"memo_id: {memo.get('id')}",
        f"engine: {json.dumps(data.get('engine', {}).get('model', ''))}",
    ]
    if memo.get("folder"):  # the Voice Memos folder, "Imported", or the recorder ("Plaud")
        front.append(f"folder: {json.dumps(memo['folder'], ensure_ascii=False)}")
    if segments:  # only for memos transcribed in parts, so other files stay as they were
        front += [f"segments: {segments['count']}",
                  f"segment_bounds_s: {json.dumps(segments['bounds_s'])}"]
    front += [
        f"threshold: {threshold}",
        f"marked_share: {s['marked_share']}",
        f"certainty: {s['certainty']}",
        f"confidence_available: {'true' if data.get('confidence_available') else 'false'}",
        f"transcribed: {data.get('created', '')}",
        "---",
        "",
        f"# {title}",
        "",
        f"*{when}, {int(duration // 60)}:{int(duration % 60):02d}, {notation.stats_line(s)}*",
        "",
        f"> {notation.READER_NOTE}",
        "",
    ]
    if segments:
        cuts = ", ".join(clock(b) for b in segments["bounds_s"][1:-1])
        front += [f"> Transcribed in {segments['count']} parts (cut at {cuts}), because the speech engine "
                  "accepts at most 300 minutes per file. Time stamps run through the whole recording.", ""]
    second = next((r for r in data.get("readers", []) if r.get("role") == "second" and not r.get("error")), None)
    skipped = next((r for r in data.get("readers", []) if r.get("role") == "second" and r.get("skipped")), None)
    if skipped and not second:
        front += [f"> Second reader skipped: {skipped['skipped']}.", ""]
    if second:
        agreement = second.get("agreement", {}).get("agreement")
        front += [f"> Second reader: {second.get('model')}, word agreement "
                  f"{100 * agreement:.0f} %. `{{a|b}}` shows where the engines differ." if agreement is not None
                  else f"> Second reader: {second.get('model')}.", ""]
    speakers = data.get("speakers") or {}
    voice = data.get("voiceprint") or {}
    if speakers:
        parts = ", ".join(f"{v['label']} {100 * v['share']:.0f} %" for v in speakers.values())
        note = (" S-labels hold within one part only: the letter is the part, and S2a and S2b are not "
                "necessarily the same person." if segments else " S-labels hold within this memo only.")
        if any(v.get("basis") == "voiceprint" for v in speakers.values()):
            scored = ", ".join(f"{v['label']} {v['voiceprint']:.2f}" for v in speakers.values()
                               if v.get("voiceprint") is not None)
            note += f" The name comes from a voiceprint match (similarity: {scored}); (?) = possible match."
        front += [f"> Speakers (share of words): {parts}.{note}", ""]
    elif voice.get("score") is not None:
        verdict = {"match": f"matches {voice['name']}", "possible": f"possibly {voice['name']} (?)",
                   "no match": f"does not match {voice['name']}"}.get(voice.get("result"), voice.get("result"))
        front += [f"> Voiceprint: {verdict} (similarity {voice['score']:.2f}).", ""]
    if not data.get("confidence_available"):
        front += ["> Warning: the engine returned no confidence values, so nothing is marked as uncertain.", ""]
    if markers:
        front += ["## Markers", ""] + [f"- **{m['label']}{' (?)' if m.get('uncertain_trigger') else ''}:** {m['text']}"
                                      for m in markers] + [""]
    front += ["## Transcript", "", text or "(no speech recognised)", ""]
    return "\n".join(front)
