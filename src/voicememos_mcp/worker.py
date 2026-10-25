"""The worker: the only component that reads the Voice Memos container.

One run (``voicememos-mcp sync``):

1. read the library database and write a snapshot to ``library.json``,
2. process queued transcription requests from the MCP server,
3. transcribe new memos automatically (if enabled); memos longer than the engine accepts are
   cut into parts, transcribed part by part (resumable) and merged (``segment.py``),
4. optionally append new transcripts to Roam,
5. record the result in ``worker.json``.

On macOS it runs from launchd, triggered by changes in the recordings folder or
the queue folder and every 30 minutes. Only this process needs Full Disk Access.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import align, notation, roam, segment
from .audio import cut_audio, find_tool, prepare_audio
from .config import Config
from .library import LibraryError, Memo, is_excluded, read_library
from .store import Store, now_iso, read_json, write_json_atomic

log = logging.getLogger(__name__)

Transcriber = Callable[[Path], list[dict]]
SecondReader = Callable[[Path], dict]


class Busy(RuntimeError):
    pass


def _bucket(item: dict) -> str:
    if item["status"] in ("transcribed", "dry_run"):
        return "transcribed"
    return "skipped" if item["status"] == "too_long" else "failed"


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
    # Allow for the longest file the engine takes (a part of a segmented memo is close to it).
    client = Soniox(key, cfg.get("SONIOX_BASE_URL"), cfg.get("SONIOX_MODEL"),
                    poll_timeout=max(1800.0, cfg.max_duration_s / 2))

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
                 second_reader: SecondReader | None = None, embedder=None, audio_loader=None,
                 audio_cutter: Callable = cut_audio):
        self.cfg = cfg
        self.store = store or Store(cfg.data_dir, cfg.output_dir)
        self.store.stamp_every_s = cfg.timestamp_every_s
        self._transcriber = transcriber
        self._second_reader = second_reader
        self._embedder = embedder
        self._audio_loader = audio_loader
        self._cut_audio = audio_cutter
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
        """The reader used after new transcriptions (``VOICEMEMOS_SECOND_READER=gemini`` or
        ``gemini-low``), or, with ``on_demand``, for explicit second-read requests even when automatic
        reading is off."""
        if self._second_reader is not None and model is None:
            return self._second_reader
        if model or on_demand or self.cfg.get("VOICEMEMOS_SECOND_READER") in ("gemini", "gemini-low"):
            reader = gemini_reader(self.cfg, model)
            if model is None:
                self._second_reader = reader
            return reader
        return None

    def wants_second_read(self, words: list[dict]) -> bool:
        """``gemini-low`` reads only memos whose primary transcript has low certainty."""
        if self.cfg.get("VOICEMEMOS_SECOND_READER") != "gemini-low":
            return True
        return notation.stats(words, self.cfg.threshold)["certainty"] == "low"

    def is_long(self, memo: Memo, words: list[dict]) -> bool:
        """Longer than ``VOICEMEMOS_SECOND_READER_MAX_MIN``: read in parts."""
        limit_ms = float(self.cfg.get("VOICEMEMOS_SECOND_READER_MAX_MIN")) * 60_000
        duration_ms = (memo.duration_s or 0) * 1000 or (words[-1].get("end_ms") or 0 if words else 0)
        return duration_ms > limit_ms

    def is_too_long(self, memo: Memo) -> bool:
        """Longer than the primary engine accepts in one file (``VOICEMEMOS_MAX_DURATION_MIN``)."""
        return bool(memo.duration_s and memo.duration_s > self.cfg.max_duration_s)

    def segment_problem(self) -> str | None:
        """Why too long memos cannot be split, or None when they can."""
        if not self.cfg.flag("VOICEMEMOS_SEGMENT_LONG"):
            return "splitting is off (VOICEMEMOS_SEGMENT_LONG=0)"
        if not find_tool("ffmpeg") or not find_tool("ffprobe"):
            return "ffmpeg/ffprobe not found (searched /opt/homebrew/bin, /usr/local/bin and PATH)"
        return None

    def too_long_reason(self, memo: Memo) -> str:
        return (f"{memo.duration_s / 60:.0f} min is longer than VOICEMEMOS_MAX_DURATION_MIN "
                f"({self.cfg.max_duration_s / 60:g} min) and {self.segment_problem()}")

    def _apply_second_reader(self, reader: SecondReader, audio: Path, memo: Memo, words: list[dict],
                             only_uncertain: bool = False) -> tuple[list[dict], dict]:
        """Run the second reader and align it. Returns (words, reader_record). Never raises.

        Long recordings are read in parts of about ``VOICEMEMOS_SECOND_READER_CHUNK_MIN``;
        with ``only_uncertain`` only parts whose primary transcript has low certainty."""
        record: dict = {"role": "second", "provider": "gemini", "created": now_iso()}
        floor = float(self.cfg.get("VOICEMEMOS_SECOND_READER_MIN_AGREEMENT"))
        try:
            if not self.is_long(memo, words):
                result = reader(audio)
                aligned, metrics = align.align(words, result["text"], self.cfg.threshold,
                                               result.get("finish_reason") in align.TRUNCATED)
                record.update(model=result.get("model"), text=result["text"], usage=result.get("usage", {}),
                              finish_reason=result.get("finish_reason"), agreement=metrics)
                if metrics["agreement"] < floor:
                    # The reader wrote something that is not this recording (seen: a made-up
                    # exercise plan). Keep its text for inspection, but do not mark anything.
                    record["error"] = (f"rejected: agreement {metrics['agreement']} below {floor}; "
                                       "the reader's text does not match the recording")
                    return align.strip_reader(words), record
                return aligned, record
            chunk_ms = int(float(self.cfg.get("VOICEMEMOS_SECOND_READER_CHUNK_MIN")) * 60_000)
            chunks = align.plan_chunks(words, chunk_ms)
            tokens, model = 0, None
            with tempfile.TemporaryDirectory(prefix="voicememos-part-") as tmp:
                for k, chunk in enumerate(chunks):
                    certainty = notation.stats(words[chunk["first"]:chunk["last"]], self.cfg.threshold)["certainty"]
                    chunk["certainty"] = certainty
                    if only_uncertain and certainty != "low":
                        chunk["text"] = None
                        continue
                    end_s = chunk["end_ms"] / 1000 if chunk["end_ms"] is not None else None
                    clip = self._cut_audio(audio, chunk["start_ms"] / 1000, end_s,
                                           Path(tmp) / f"part{k + 1}{audio.suffix or '.m4a'}")
                    try:
                        result = reader(clip)
                    except Exception as exc:  # noqa: BLE001 - one refused part must not lose the others
                        log.warning("Second reader failed for part %d of %s: %s", k + 1, memo.id, exc)
                        chunk.update(text=None, failed=str(exc)[:200])
                        continue
                    chunk.update(text=result["text"], finish_reason=result.get("finish_reason"))
                    _, part_metrics = align.align(words[chunk["first"]:chunk["last"]], result["text"],
                                                  self.cfg.threshold,
                                                  result.get("finish_reason") in align.TRUNCATED)
                    if part_metrics["agreement"] < floor:
                        chunk.update(text=None, rejected_text=result["text"],
                                     rejected=f"agreement {part_metrics['agreement']} below {floor}")
                    tokens += (result.get("usage") or {}).get("totalTokenCount", 0)
                    model = model or result.get("model")
            if not any(c.get("text") is not None for c in chunks):
                if any(c.get("rejected") for c in chunks):
                    record.update(chunks=chunks, error="rejected: no part matched the recording")
                    return align.strip_reader(words), record
                if any(c.get("failed") for c in chunks):
                    record["error"] = "every part failed: " + next(c["failed"] for c in chunks if c.get("failed"))
                    return words, record
                record["error"] = "skipped: no part with low certainty"
                return words, record
            aligned, metrics = align.align_chunks(words, chunks, self.cfg.threshold)
            record.update(model=model, chunks=chunks, usage={"totalTokenCount": tokens}, agreement=metrics)
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
        model = self.cfg.get("VOICEMEMOS_VOICEPRINT_MODEL")
        if voiceprint.get("model") != model:
            return speakers, {"name": voiceprint.get("name"), "result": "error",
                              "error": f"the voiceprint was made with {voiceprint.get('model')}, but the "
                                       f"configured model is {model}; enroll again"}
        try:
            return identify(self.load_audio(audio), words, speakers, voiceprint, self.embedder())
        except Exception as exc:  # noqa: BLE001 - the transcript must survive
            log.exception("Voiceprint identification failed")
            return speakers, {"name": voiceprint.get("name"), "result": "error", "error": str(exc)[:200]}

    def _identify_parts(self, audio: Path, words: list[dict], bounds_s: list[float]) -> tuple[dict, dict | None]:
        """The voiceprint for a memo transcribed in parts: every part is scored on its own, because
        the engine's speaker labels hold within one part only. Never raises."""
        from .voiceprint import Voiceprint, as_reader, identify

        speakers = segment.speaker_labels(words)
        voiceprint = Voiceprint(self.store.voiceprint_path).load()
        if not voiceprint:
            return speakers, None
        name, model = voiceprint.get("name"), self.cfg.get("VOICEMEMOS_VOICEPRINT_MODEL")
        if voiceprint.get("model") != model:
            return speakers, {"name": name, "result": "error",
                              "error": f"the voiceprint was made with {voiceprint.get('model')}, but the "
                                       f"configured model is {model}; enroll again"}
        try:
            read, embed = as_reader(self.load_audio(audio)), self.embedder()
            parts = []
            for k, part_words in enumerate(segment.split_words(words, bounds_s)):
                local = {key: v for key, v in speakers.items() if key.startswith(f"{k + 1}:")}
                # Word times are absolute; a read from 0 (the fallback without word timings) means this part.
                in_part = lambda start, duration, a=bounds_s[k]: read(max(start, a), duration)  # noqa: E731
                updated, summary = identify(in_part, part_words, local if len(local) > 1 else {}, voiceprint, embed)
                if len(local) > 1:
                    speakers.update(updated)
                elif len(local) == 1 and summary.get("score") is not None:
                    # One speaker in this part, others in other parts: the part's score is that speaker's.
                    key = next(iter(local))
                    speakers[key] = dict(speakers[key], voiceprint=summary["score"])
                    if summary["result"] in ("match", "possible"):
                        speakers[key].update(label=name if summary["result"] == "match" else f"{name} (?)",
                                             basis="voiceprint")
                        summary["speaker"] = key
                parts.append({"part": k + 1, **{x: summary[x] for x in ("result", "score", "scores", "speaker")
                                                if summary.get(x) is not None}})
        except Exception as exc:  # noqa: BLE001 - the transcript must survive
            log.exception("Voiceprint identification failed")
            return speakers, {"name": name, "result": "error", "error": str(exc)[:200]}
        summary = {"name": name, "thresholds": voiceprint["thresholds"], "parts": parts}
        if speakers:  # the owner found in some part
            found = [p for p in parts if p.get("speaker")]
            match = next((p for p in found if p["result"] == "match"), found[0] if found else None)
            summary.update(result=match["result"] if match else "no match",
                           **({"speaker": match["speaker"]} if match else {}))
            return speakers, summary
        scores = [p["score"] for p in parts if p.get("score") is not None]
        if not scores:
            return speakers, {**summary, "score": None, "result": "too little speech"}
        score = round(sum(scores) / len(scores), 4)
        thresholds = voiceprint["thresholds"]
        result = "match" if score >= thresholds["match"] else "possible" if score >= thresholds["maybe"] else "no match"
        if result == "match" and any(p["result"] != "match" for p in parts):
            result = "possible"  # not every part sounds like the owner
        return speakers, {**summary, "score": score, "result": result}

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
        previous = Voiceprint(self.store.voiceprint_path).load() or {}
        if previous.get("model") == result["model"] and previous.get("others"):
            result["others"] = previous["others"]  # known other voices stay valid with the same model
        Voiceprint(self.store.voiceprint_path).save(result)
        return Voiceprint(self.store.voiceprint_path).summary()

    def add_other_voice(self, memo: Memo, speaker: str | None, name: str) -> dict:
        """Store a known other person's voice ("not me") from one speaker of a transcribed memo."""
        from .voiceprint import Voiceprint, other_voice

        store = Voiceprint(self.store.voiceprint_path)
        data = store.load()
        if not data:
            raise RuntimeError("enroll your own voiceprint first")
        if data.get("model") != self.cfg.get("VOICEMEMOS_VOICEPRINT_MODEL"):
            raise RuntimeError("the voiceprint was made with another model; enroll again first")
        entry = self.state.get(memo.id, {})
        transcript = self.store.load_transcript(entry["json"]) if entry.get("ok") else None
        if not transcript:
            raise RuntimeError(f"memo {memo.short_id} is not transcribed yet")
        labels = transcript.get("speakers") or {}
        if labels and not speaker:
            raise RuntimeError(f"memo {memo.short_id} has {len(labels)} speakers; say which one (S1, S2, ...)")
        if speaker and speaker not in labels:
            raise RuntimeError(f"memo {memo.short_id} has no speaker S{speaker}")
        if not memo.local or not memo.filename:
            raise RuntimeError(f"memo {memo.short_id} is not downloaded")
        with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
            samples = self.load_audio(prepare_audio(self.cfg.recordings_dir / memo.filename, Path(tmp)))
            voice = other_voice(name, samples, transcript.get("words", []), speaker, self.embedder())
        voice.update(memo_id=memo.id, speaker=speaker)
        others = [o for o in data.get("others", []) if (o.get("memo_id"), o.get("speaker")) != (memo.id, speaker)]
        data["others"] = others + [voice]
        store.save(data)
        return store.summary()

    def remove_other_voice(self, name: str) -> dict:
        from .voiceprint import Voiceprint

        store = Voiceprint(self.store.voiceprint_path)
        data = store.load()
        if not data:
            raise RuntimeError("no voiceprint enrolled")
        kept = [o for o in data.get("others", []) if o.get("name") != name]
        if len(kept) == len(data.get("others", [])):
            raise RuntimeError(f"no known other voice named '{name}'")
        data["others"] = kept
        store.save(data)
        return store.summary()

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
                if data.get("segments"):
                    speakers, summary = self._identify_parts(audio, words, data["segments"]["bounds_s"])
                else:
                    speakers, summary = self._identify(audio, words, notation.speaker_labels(words))
            if summary is None:
                return {**item, "status": "failed", "error": "no voiceprint enrolled"}
            if summary.get("result") == "error":
                return {**item, "status": "failed", "error": summary.get("error")}
            data["speakers"], data["voiceprint"] = speakers, summary
            result = self.store.rewrite_transcript(entry["json"], data, self.cfg.threshold, self.cfg.run_min,
                                                   self.cfg.pause_ms, self.cfg.marker_spec)
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
        if self.is_too_long(memo):
            if self.segment_problem():
                return "too_long"
            plan = entry.get("segments")
            if plan:  # parts in progress; exhausted parts make the memo fail like a single file would
                exhausted = any(p["status"] != "done" and p.get("attempts", 0) >= self.cfg.max_attempts
                                for p in plan["parts"])
                return "failed" if exhausted else "segmenting"
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
            if not entry.get("ok") and entry.get("segments"):
                parts = entry["segments"]["parts"]
                item["segments"] = {"done": sum(p["status"] == "done" for p in parts), "total": len(parts)}
                failed = next((p for p in parts if p.get("error") and p["status"] != "done"), None)
                if failed and not item.get("error"):
                    item["error"] = failed["error"]
            elif item["status"] == "too_long":
                item["error"] = self.too_long_reason(memo)
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

        if self.is_too_long(memo) and self.segment_problem():
            reason = self.too_long_reason(memo)
            log.warning("Not transcribing %s: %s", memo.id, reason)
            return {**item, "status": "too_long", "error": reason}
        try:
            if self.is_too_long(memo):
                if dry_run:
                    return self._plan_only(memo, path, item)
                return self._transcribe_segmented(memo, path, fingerprint, entry, item, force, push_roam)
            with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
                audio = prepare_audio(path, Path(tmp))
                tokens = self.transcriber(audio)
                words, had_confidence = notation.tokens_to_words(tokens)
                if dry_run:
                    text = "\n\n".join(notation.render_blocks(words, self.cfg.threshold, self.cfg.run_min,
                                                              self.cfg.pause_ms, notation.speaker_labels(words)))
                    return {**item, "status": "dry_run", "stats": notation.stats(words, self.cfg.threshold),
                            "text": text, "confidence_available": had_confidence}
                return self._finish(memo, item, entry, fingerprint, audio, words, had_confidence, tokens, push_roam)
        except Exception as exc:  # noqa: BLE001 - one failing memo must not stop the run
            log.exception("Transcription failed for %s", memo.id)
            if not dry_run:
                current = self.state.get(memo.id, entry)  # keeps the progress of a memo in parts
                self.state[memo.id] = {**current, "ok": False, "fp": fingerprint,
                                       "attempts": current.get("attempts", 0) + 1, "error": str(exc)[:300],
                                       "at": now_iso()}
            return {**item, "status": "failed", "error": str(exc)[:300]}

    def _finish(self, memo: Memo, item: dict, entry: dict, fingerprint: str, audio: Path, words: list[dict],
                had_confidence: bool, tokens: list[dict], push_roam: bool, segments: dict | None = None) -> dict:
        """Second reader, voiceprint, transcript files and state for a fresh primary transcript."""
        second_record = None
        reader = self.second_reader()
        if reader is not None:
            total_min = float(self.cfg.get("VOICEMEMOS_SECOND_READER_MAX_TOTAL_MIN") or 0)
            duration_min = (memo.duration_s or 0) / 60
            if total_min and duration_min > total_min:
                reason = (f"the recording is {duration_min:.0f} min, longer than "
                          f"VOICEMEMOS_SECOND_READER_MAX_TOTAL_MIN ({total_min:g} min)")
                log.warning("Second reader skipped for %s: %s", memo.id, reason)
                second_record = {"role": "second", "provider": "gemini", "created": now_iso(),
                                 "skipped": reason, "error": f"skipped: {reason}"}
            elif self.is_long(memo, words) or self.wants_second_read(words):
                only_uncertain = self.cfg.get("VOICEMEMOS_SECOND_READER") == "gemini-low"
                words, second_record = self._apply_second_reader(reader, audio, memo, words, only_uncertain)
        if segments:
            speakers, voice = self._identify_parts(audio, words, segments["bounds_s"])
        else:
            speakers, voice = self._identify(audio, words, notation.speaker_labels(words))
        engine = {"provider": "soniox", "model": self.cfg.get("SONIOX_MODEL"),
                  "language_hints": self.cfg.languages}
        extra = {"language": align.dominant_language(words), "speakers": speakers, "voiceprint": voice,
                 "readers": [{"role": "primary", **engine, "created": now_iso()}]
                 + ([second_record] if second_record else [])}
        if segments:
            extra["segments"] = segments
        result = self.store.write_transcript(
            memo.to_dict(), words, engine, had_confidence, self.cfg.threshold,
            self.cfg.run_min, self.cfg.pause_ms, self.cfg.marker_spec, source_tokens=tokens, extra=extra,
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
        if segments:
            item["segments"] = segments["count"]
        if push_roam and self.cfg.flag("VOICEMEMOS_ROAM_ENABLED") and not new_entry.get("roam"):
            self._push_roam(memo, words, result, new_entry, item)
        return item

    # ------------------------------------------------------------------ memos in parts
    def _plan(self, memo: Memo, audio: Path) -> list[float]:
        window = float(self.cfg.get("VOICEMEMOS_SEGMENT_SILENCE_WINDOW_MIN") or 0) * 60
        return segment.plan_bounds(memo.duration_s, self.cfg.segment_max_s, window, audio=audio)

    def _plan_only(self, memo: Memo, path: Path, item: dict) -> dict:
        """Dry run of a memo in parts: where it would be cut, without calling the engine."""
        with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
            bounds = self._plan(memo, prepare_audio(path, Path(tmp)))
        cuts = ", ".join(segment.clock(b) for b in bounds[1:-1])
        return {**item, "status": "dry_run", "stats": notation.stats([], self.cfg.threshold), "segments": bounds,
                "text": f"(not transcribed in a dry run: {len(bounds) - 1} parts, cut at {cuts})"}

    def _transcribe_segmented(self, memo: Memo, path: Path, fingerprint: str, entry: dict, item: dict,
                              force: bool, push_roam: bool) -> dict:
        """Transcribe a too long memo in parts. Each part is its own job; finished parts are kept
        (raw tokens in ``tmp/<memo id>/``), so a failed part is the only one tried again."""
        workdir = self.store.tmp_dir / memo.id
        plan = entry.get("segments")
        if force or not plan or plan.get("fp") != fingerprint:
            shutil.rmtree(workdir, ignore_errors=True)  # a new plan: earlier parts no longer fit
            plan = None
        workdir.mkdir(parents=True, exist_ok=True)
        audio = prepare_audio(path, workdir)
        if plan is None:
            bounds = self._plan(memo, audio)
            plan = {"fp": fingerprint, "bounds_s": bounds, "created": now_iso(),
                    "parts": [{"status": "pending", "attempts": 0} for _ in bounds[1:]]}
            log.info("Memo %s is %.0f min, longer than %g min: %d parts, cut at %s", memo.id,
                     memo.duration_s / 60, self.cfg.max_duration_s / 60, len(bounds) - 1,
                     ", ".join(segment.clock(b) for b in bounds[1:-1]))
        self.state[memo.id] = {**entry, "segments": plan}
        self.store.write_state(self.state)
        bounds, n = plan["bounds_s"], len(plan["parts"])
        for k, part in enumerate(plan["parts"]):
            tokens_file = workdir / f"part{k + 1}.tokens.json"
            if part["status"] == "done" and tokens_file.exists():
                continue
            end_s = bounds[k + 1] if k + 1 < n else None  # the last part runs to the end of the file
            job_id = hashlib.sha1(f"{memo.id}:{fingerprint}:{k}".encode()).hexdigest()[:12]
            job = self.store.read_job(job_id) or {"job_id": job_id, "created": now_iso()}
            job.update(status="running", started=now_iso(), total=1, done=0, failed=0,
                       request={"action": "transcribe_part", "memos": [memo.id], "part": k + 1, "of": n,
                                "start_s": bounds[k], "end_s": bounds[k + 1]},
                       items=[{"short_id": memo.short_id, "title": memo.title, "recorded": memo.recorded,
                               "status": "running"}])
            job.pop("error", None)
            self.store.write_job(job)
            try:
                clip = workdir / f"part{k + 1}.m4a"
                if not clip.exists():
                    partial = workdir / f"part{k + 1}.partial.m4a"
                    segment.cut_part(audio, bounds[k], end_s, partial)
                    partial.replace(clip)
                tokens = self.transcriber(clip)
                write_json_atomic(tokens_file, tokens)
                part.update(status="done", at=now_iso())
                part.pop("error", None)
                job.update(status="done", done=1)
            except Exception as exc:  # noqa: BLE001 - the other parts are kept
                log.exception("Part %d/%d of %s failed", k + 1, n, memo.id)
                part.update(status="failed", attempts=part.get("attempts", 0) + 1, error=str(exc)[:300], at=now_iso())
                job.update(status="done", done=1, failed=1, error=str(exc)[:300])
            job["items"][0].update(status="transcribed" if part["status"] == "done" else "failed",
                                   error=part.get("error"))
            job["finished"] = now_iso()
            self.store.write_job(job)
            self.store.write_state(self.state)
        missing = [k for k, p in enumerate(plan["parts"]) if p["status"] != "done"]
        if missing:
            return {**item, "status": "failed", "segments": n,
                    "error": f"part {missing[0] + 1}/{n} failed: {plan['parts'][missing[0]].get('error')} "
                             "(the next run tries only the missing parts)"}
        parts = [{"start_s": bounds[k], "tokens": read_json(workdir / f"part{k + 1}.tokens.json")}
                 for k in range(n)]
        words, had_confidence, tokens, ranges = segment.merge_parts(parts)
        segments = {"count": n, "bounds_s": bounds, "source_token_ranges": ranges}
        item = self._finish(memo, item, entry, fingerprint, audio, words, had_confidence, tokens, push_roam, segments)
        shutil.rmtree(workdir, ignore_errors=True)  # kept on failure, for diagnosis
        return item

    def second_read_memo(self, memo: Memo, model: str | None = None, redo: bool = False,
                         only_uncertain: bool = False) -> dict:
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
        if only_uncertain and not self.is_long(memo, data.get("words", [])):
            certainty = notation.stats(data.get("words", []), self.cfg.threshold)["certainty"]
            if certainty != "low":
                return {**item, "status": "skipped", "error": f"certainty {certainty}"}
        if not memo.local or not memo.filename:
            return {**item, "status": "not_downloaded"}
        try:
            reader = self.second_reader(model, on_demand=True)
            if reader is None:
                return {**item, "status": "failed", "error": "no second reader configured"}
            with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
                audio = prepare_audio(self.cfg.recordings_dir / memo.filename, Path(tmp))
                words, record = self._apply_second_reader(reader, audio, memo, data.get("words", []),
                                                          only_uncertain)
        except Exception as exc:  # noqa: BLE001
            return {**item, "status": "failed", "error": str(exc)[:300]}
        if str(record.get("error", "")).startswith("skipped"):
            return {**item, "status": "skipped", "error": record["error"]}
        rejected = str(record.get("error", "")).startswith("rejected")
        if record.get("error") and not rejected:
            return {**item, "status": "failed", "error": record["error"]}
        readers = [r for r in data.get("readers", []) if r.get("role") != "second"] + [record]
        if not any(r.get("role") == "primary" for r in readers):
            readers.insert(0, {"role": "primary", **data.get("engine", {})})
        data["readers"] = readers
        data["words"] = words
        data.setdefault("language", align.dominant_language(words))
        result = self.store.rewrite_transcript(entry["json"], data, self.cfg.threshold, self.cfg.run_min,
                                               self.cfg.pause_ms, self.cfg.marker_spec)
        entry.update(stats=result["stats"], markers=result["markers"], language=data["language"],
                     speakers=len(data.get("speakers") or {}))
        if rejected:
            entry["second_reader"] = {"model": record.get("model"), "rejected": record["error"],
                                      "at": record["created"]}
            self.state[memo.id] = entry
            return {**item, "status": "rejected", "error": record["error"], "markdown": result["markdown"]}
        entry["second_reader"] = {"model": record.get("model"), "agreement": record["agreement"]["agreement"],
                                  "at": record["created"]}
        self.state[memo.id] = entry
        return {**item, "status": "second_read", "agreement": record["agreement"], "markdown": result["markdown"]}

    def realign_memo(self, memo: Memo) -> dict:
        """Re-align the stored second reading with the current rules; no new API call."""
        item = {"id": memo.id, "short_id": memo.short_id, "title": memo.title, "recorded": memo.recorded}
        entry = self.state.get(memo.id, {})
        data = self.store.load_transcript(entry["json"]) if entry.get("ok") else None
        if data is None:
            return {**item, "status": "skipped", "error": "no transcript"}
        second = next((r for r in data.get("readers", []) if r.get("role") == "second"
                       and (r.get("text") or r.get("chunks"))), None)
        if second is None:
            return {**item, "status": "skipped", "error": "no second reading"}
        floor = float(self.cfg.get("VOICEMEMOS_SECOND_READER_MIN_AGREEMENT"))
        primary = data.get("words", [])
        if second.get("chunks"):
            for chunk in second["chunks"]:
                if chunk.get("rejected"):  # give a rejected part another look under the current floor
                    chunk["text"] = chunk.pop("rejected_text", None)
                    chunk.pop("rejected")
                if chunk.get("text") is None:
                    continue
                _, m = align.align(primary[chunk["first"]:chunk["last"]], chunk["text"], self.cfg.threshold,
                                   chunk.get("finish_reason") in align.TRUNCATED)
                if m["agreement"] < floor:
                    chunk.update(rejected_text=chunk["text"], text=None, rejected=f"agreement {m['agreement']} below {floor}")
            words, metrics = align.align_chunks(primary, second["chunks"], self.cfg.threshold)
            ok = metrics["chunks_read"] > 0
        else:
            words, metrics = align.align(primary, second["text"], self.cfg.threshold,
                                         second.get("finish_reason") in align.TRUNCATED)
            ok = metrics["agreement"] >= floor
        second["agreement"] = metrics
        if ok:
            second.pop("error", None)
        else:
            words = align.strip_reader(primary)
            second["error"] = f"rejected: agreement {metrics['agreement']} below {floor}"
        data["words"] = words
        result = self.store.rewrite_transcript(entry["json"], data, self.cfg.threshold, self.cfg.run_min,
                                               self.cfg.pause_ms, self.cfg.marker_spec)
        entry.update(stats=result["stats"], markers=result["markers"])
        entry["second_reader"] = ({"model": second.get("model"), "agreement": metrics["agreement"],
                                   "at": second.get("created")} if ok else
                                  {"model": second.get("model"), "rejected": second["error"], "at": second.get("created")})
        self.state[memo.id] = entry
        return {**item, "status": "realigned" if ok else "rejected", "agreement": metrics["agreement"]}

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
            action = request.get("action")
            if action == "add_other":
                for memo in targets:
                    job["voiceprint"] = self.add_other_voice(memo, request.get("speaker"), request["name"])
            elif action == "remove_other":
                job["voiceprint"] = self.remove_other_voice(request["name"])
            else:
                job["voiceprint"] = self.enroll_voiceprint(targets, request.get("name")
                                                           or self.cfg.get("VOICEMEMOS_VOICEPRINT_NAME"))
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("Voiceprint job failed")
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
            if action in ("enroll", "add_other", "remove_other"):
                self._run_enroll_job(job, request, memos)
                jobs.append(job)
                continue
            targets = self._resolve_request(request, memos)
            if action == "identify":
                targets = [m for m in targets if self.state.get(m.id, {}).get("ok")]
            if request.get("action") == "second_read":
                targets = [m for m in targets if self.state.get(m.id, {}).get("ok")]
                if not request.get("redo") and not request.get("realign"):
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
                elif second and request.get("realign"):
                    item = self.realign_memo(memo)
                elif second:
                    item = self.second_read_memo(memo, request.get("model"), bool(request.get("redo")),
                                                 bool(request.get("only_uncertain")))
                else:
                    item = self.transcribe_memo(memo, force=bool(request.get("redo")), push_roam=push)
                job["items"].append({k: item.get(k) for k in ("short_id", "title", "recorded", "status",
                                                             "error", "markdown")})
                job["done"] += 1
                job["failed"] += item["status"] == "failed"
                if second and item.get("agreement") is not None:
                    agreement = item["agreement"]
                    job["items"][-1]["agreement"] = agreement.get("agreement") if isinstance(agreement, dict) else agreement
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
            if self._status(memo) not in ("pending", "segmenting"):
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
                        limit: int | None = None, model: str | None = None, redo: bool = False,
                        only_uncertain: bool = False, realign: bool = False) -> dict:
        """Add the second reader to existing transcripts (synchronously, under the worker lock).

        With ``realign`` the stored readings are aligned again with the current rules instead."""
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
                    if realign:
                        items.append(self.realign_memo(memo))
                    else:
                        items.append(self.second_read_memo(memo, model, redo, only_uncertain))
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
                         "jobs": [], "transcribed": [], "failed": [], "skipped": []}
        problem = self.segment_problem()
        if problem and self.cfg.flag("VOICEMEMOS_SEGMENT_LONG"):
            log.error("Memos longer than %g min cannot be transcribed: %s",
                      self.cfg.max_duration_s / 60, problem)
            summary["warnings"] = [f"long memos cannot be split: {problem}"]
        if not dry_run:
            self.store.write_library(self._snapshot(memos))
            self.store.write_worker({"ok": True, "running": True, "started": started, "library_readable": True,
                                     "memos": len(memos)})

        if only:
            targets = self._resolve_request({"memos": only}, memos)
            for memo in targets[: limit or None]:
                item = self.transcribe_memo(memo, force=redo, dry_run=dry_run)
                summary[_bucket(item)].append(item)
        else:
            if not dry_run:
                summary["jobs"] = [{"job_id": j["job_id"], "total": j.get("total"), "failed": j.get("failed")}
                                   for j in self.process_queue(memos)]
            count = 0
            for memo in self.auto_candidates(memos):
                if limit and count >= limit:
                    break
                item = self.transcribe_memo(memo, push_roam=True, dry_run=dry_run)
                summary[_bucket(item)].append(item)
                count += 1
                if not dry_run:
                    self.store.write_state(self.state)
                    if count % 5 == 0:  # let the MCP tools see progress during long runs
                        self.store.write_library(self._snapshot(memos))

            # Memos that will not be transcribed at all must not disappear silently.
            for memo in memos:
                if not self._excluded(memo) and memo.local and self._status(memo) == "too_long":
                    reason = self.too_long_reason(memo)
                    log.warning("Not transcribing %s: %s", memo.id, reason)
                    summary["skipped"].append({"id": memo.id, "short_id": memo.short_id, "title": memo.title,
                                               "recorded": memo.recorded, "status": "too_long", "error": reason})

        if not dry_run:
            self.store.write_state(self.state)
            self.store.write_library(self._snapshot(memos))
            summary["finished"] = now_iso()
            self.store.write_worker({k: v for k, v in summary.items() if k not in ("transcribed", "failed")}
                                    | {"transcribed_count": len(summary["transcribed"]),
                                       "failed_count": len(summary["failed"])})
        return summary
