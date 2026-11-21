"""Taking a stretch of time out of one recording, for good: ``voicememos-mcp redact``.

For a recording that caught something that must not be kept (a third party's matters in the middle of a
meeting) when the rest should stay. The range leaves every form the pipeline keeps:

- the audio: silent in place (``opus.py``), as long as before, so every other time stamp still points at
  the same sound. Only Ogg Opus files in the import folder (a recorder's, an imported file); a recording
  in Apple's Voice Memos library is never edited;
- the sidecar: the range's words, the engine's raw tokens, the markers found there, and the text of every
  part of the second reader's reading that overlaps the range (that text has no times);
- the transcript: rendered again from the sidecar, with one line where the range was,
  ``[poistettu 21.11.2026: 1:53:15–1:58:57]``: the day and the bounds, no reason and no content;
- the state: the audio's fingerprint (so the recording is not transcribed again), the markers kept in
  ``state.json``, its backup and ``library.json``, the engine's cached result, the report samples.

``DATA_DIR/redactions.json`` records the ranges (memo, bounds, day). It is written first and never
removed, and everything that later writes a transcript or reads the audio honours it: a transcription
from scratch drops the range and renders its line (``Store.write_transcript``), every re-rendering keeps
the line (it is rendered from the sidecar's ``redactions``), and an audio file that was fetched again is
silenced before anything reads it (``ensure_audio``).

Every step checks what is there and does what is missing, so a run cut short is finished by running the
same command again, and a second run changes nothing. Files are replaced through a temporary file in the
same directory; no copy of the original is left.

Nothing here prints, logs or returns words of a recording, with one exception made for the person who
gives the command: ``plan`` returns the three words before and after the range, which are outside it,
so the bounds can be checked without reading what is between them.
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import difflib
import fcntl
import logging
import os
import re
import shutil
import time
from pathlib import Path

from . import notation, opus
from .store import SCHEMA, Store, read_json, write_json_atomic, write_text_atomic

log = logging.getLogger(__name__)
EDGE_WORDS = 3
LOCK_WAIT_S = 600.0
REPORT_PAD_MS = 30_000  # a report sample shows the words around a disagreement


class RedactError(RuntimeError):
    """The range cannot be taken out as asked. The message never quotes a recording."""


def _clock(ms: int) -> str:
    sec = int(ms // 1000)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


@contextlib.contextmanager
def worker_lock(store: Store, wait_s: float = LOCK_WAIT_S):
    """The worker's own run lock, waited for: a run may be in the middle of a recording."""
    path = store.data_dir / "worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        until = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= until:
                    raise RedactError(f"the worker has held its lock for over {int(wait_s)} s; nothing was changed, "
                                      "try again when it has finished") from None
                time.sleep(1.0)
        yield


# ---------------------------------------------------------------------------- the recording's files
class Recording:
    def __init__(self, cfg, store: Store, memo_id: str):
        self.cfg, self.store, self.memo_id = cfg, store, str(memo_id)
        self.state = store.read_state()
        entry = self.state.get(self.memo_id)
        if not isinstance(entry, dict) or not entry.get("json"):
            raise RedactError(f"no transcript on record for memo {self.memo_id}")
        self.entry = entry
        sidecar = store._resolve(entry["json"])
        data = read_json(sidecar) if sidecar is not None else None
        if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("memo"), dict):
            raise RedactError(f"the transcript's data file of memo {self.memo_id} cannot be read")
        self.sidecar, self.data = sidecar, data
        self.md = sidecar.with_suffix(".md")
        name = str(data["memo"].get("filename") or "")
        candidate = cfg.import_dir / name
        self.audio: Path | None = None
        self.audio_problem: str | None = None
        if not name or "/" in name or name.startswith("."):
            self.audio_problem = "the transcript names no audio file"
        elif not candidate.is_file():
            self.audio_problem = ("the audio is not in the import folder: a recording in the Voice Memos library is "
                                  "never edited, and a file that is gone cannot be")
        elif not opus.is_opus(candidate):
            self.audio_problem = f"the audio is not Ogg Opus ({candidate.suffix or 'no suffix'}): only that can be silenced in place"
        else:
            self.audio = candidate

    def render(self, data: dict) -> str:
        """The transcript as ``Store.rewrite_transcript`` would write it, without writing."""
        cfg = self.cfg
        data = copy.deepcopy(data)
        words = data.get("words", [])
        text = "\n\n".join(notation.render_blocks(words, cfg.threshold, cfg.run_min, cfg.pause_ms, data.get("speakers")))
        data["stats"] = notation.stats(words, cfg.threshold)
        data["markers"] = notation.extract_markers(text, cfg.marker_spec, words, cfg.threshold)
        data["words"] = notation.annotate(words, cfg.threshold)
        from .store import render_markdown

        return render_markdown(data, cfg.threshold, cfg.run_min, cfg.pause_ms, cfg.timestamp_every_s)


# ---------------------------------------------------------------------------- the bounds
def _timed(words: list[dict]) -> None:
    if any(w.get("start_ms") is None or w.get("end_ms") is None for w in words):
        raise RedactError("the recording has words without a time: a range cannot be placed in it")


def _inside(words: list[dict], start_ms: int, end_ms: int) -> list[int]:
    return [i for i, w in enumerate(words) if start_ms <= w["start_ms"] < end_ms]


def snap(data: dict, from_ms: int, to_ms: int) -> dict:
    """The asked range moved onto word boundaries, outward: its start to the end of the word before, its end
    to the start of the word after. A word a bound falls inside goes with the range, and a group of words the
    second reader gave one alternative for is never split."""
    words = data.get("words") or []
    if not words:
        raise RedactError("the recording has no words")
    _timed(words)
    if not to_ms > from_ms >= 0:
        raise RedactError("a range needs a start before its end")
    picked = [i for i, w in enumerate(words) if w["end_ms"] > from_ms and w["start_ms"] < to_ms]
    if not picked:
        raise RedactError(f"no words between {_clock(from_ms)} and {_clock(to_ms)}: nothing to take out there")
    first, last = picked[0], picked[-1]
    duration_ms = int(round(float((data.get("memo") or {}).get("duration_s") or 0) * 1000))
    for _ in range(len(words)):
        while first > 0 and words[first].get("in_span"):  # the group's first word carries its alternative
            first -= 1
        for back in range(last, -1, -1):  # the group that reaches over the last word, if any
            n = max((words[back].get("span") or {}).get("n", 0), 0)
            if n:
                last = max(last, back + n - 1)
            if not words[back].get("in_span"):
                break
        last = min(last, len(words) - 1)
        start_ms = 0 if first == 0 else min(words[first - 1]["end_ms"], words[first]["start_ms"])
        end_ms = words[last + 1]["start_ms"] if last + 1 < len(words) else max(duration_ms, words[last]["end_ms"] + 1)
        inside = _inside(words, start_ms, end_ms)
        if inside == list(range(first, last + 1)):
            break
        first, last = min(inside[0], first), max(inside[-1], last)  # words with the same times as a bound word
    else:
        raise RedactError("the words' times are not in order around the range")
    return {"first": first, "last": last, "start_ms": int(start_ms), "end_ms": int(end_ms),
            "whole": first == 0 and last == len(words) - 1}


# ---------------------------------------------------------------------------- the sidecar
def _token_range(data: dict, first: int, last: int, start_ms: int, end_ms: int) -> list[int]:
    """Which raw tokens make the words first..last: by the pipeline's own grouping when it gives exactly the
    sidecar's words (a token that starts with a space starts a word), else by time."""
    tokens = data.get("source_tokens")
    if not isinstance(tokens, list):
        return []
    words = data.get("words") or []
    owner: list[int | None] = []
    count = 0
    for token in tokens:
        text = token.get("text", "") if isinstance(token, dict) else ""
        stripped = text.strip()
        if not text or token.get("translation_status") == "translation" or (stripped.startswith("<") and stripped.endswith(">")):
            owner.append(None)
            continue
        punctuation = bool(stripped) and all(ch in ".,!?;:…-–—\"'()" for ch in stripped)
        if count == 0 or (text[0].isspace() and not punctuation):
            count += 1
        owner.append(count - 1)
    starts: dict[int, object] = {}
    for k, o in enumerate(owner):
        if o is not None:
            starts.setdefault(o, tokens[k].get("start_ms"))
    if count == len(words) and all(starts.get(i) == w.get("start_ms") for i, w in enumerate(words)):
        picked, previous = [], None
        for k, o in enumerate(owner):
            previous = o if o is not None else previous  # a control token goes with the word before it
            if previous is not None and first <= previous <= last:
                picked.append(k)
        return picked
    picked, gone = [], False
    for k, token in enumerate(tokens):
        at = token.get("start_ms") if isinstance(token, dict) else None
        gone = (start_ms <= at < end_ms) if at is not None else gone
        if gone:
            picked.append(k)
    return picked


def _resume(cfg, data: dict, last: int) -> dict:
    """What the renderer needs to stamp the paragraphs after the range as they were (``notation.layout``)."""
    words = data.get("words") or []
    if not cfg.timestamp_every_s or last + 1 >= len(words):
        return {}
    lay = notation.layout(words, cfg.threshold, cfg.run_min, cfg.pause_ms, data.get("speakers"), cfg.timestamp_every_s,
                          data.get("redactions"))
    after = words[last + 1]["start_ms"]
    for k, start in enumerate(lay["starts"]):
        if start is not None and start >= after:
            return {"stamp_last_ms": lay["stamp_before"][k], "block_start": start == after}
    return {"block_start": False}


def without(cfg, data: dict, start_ms: int, end_ms: int, day: str) -> tuple[dict, dict]:
    """A copy of the sidecar without the range, and the counts of what went. With nothing left in the range
    and the range on its record, the copy equals the original."""
    out = copy.deepcopy(data)
    words = out.get("words") or []
    _timed(words)
    inside = _inside(words, start_ms, end_ms)
    counts = {"words": len(inside), "source_tokens": 0, "markers": 0, "reader_parts": 0, "reader_insertions": 0}
    record = [r for r in out.get("redactions") or [] if isinstance(r, dict)]
    mine = next((r for r in record if r.get("start_ms") == start_ms and r.get("end_ms") == end_ms), None)
    if mine is None:
        mine = {"start_ms": start_ms, "end_ms": end_ms, "date": str(day)[:10]}
        record.append(mine)
    if inside:
        first, last = inside[0], inside[-1]
        if inside != list(range(first, last + 1)):
            raise RedactError("the words of the range are not one run")
        if words[first].get("in_span") or (last + 1 < len(words) and words[last + 1].get("in_span")):
            raise RedactError("the range would split a group of words read as one: plan it again")
        resume = _resume(cfg, out, last)
        if resume:
            mine["resume"] = resume
        picked = _token_range(out, first, last, start_ms, end_ms)
        counts["source_tokens"] = len(picked)
        if picked:
            gone = set(picked)
            out["source_tokens"] = [t for k, t in enumerate(out["source_tokens"]) if k not in gone]
            ranges = (out.get("segments") or {}).get("source_token_ranges")
            if isinstance(ranges, list):  # a recording transcribed in parts: its parts' token ranges move down
                out["segments"]["source_token_ranges"] = [[x - sum(1 for g in picked if g < x) for x in pair] for pair in ranges]
        n = len(inside)
        # What the second reader heard between a kept word and the range belongs to the range.
        if first > 0 and words[first - 1].get("insert_after"):
            counts["reader_insertions"] += len(words[first - 1].pop("insert_after"))
        if last + 1 < len(words) and words[last + 1].get("insert_before"):
            counts["reader_insertions"] += len(words[last + 1].pop("insert_before"))
        out["words"] = words[:first] + words[last + 1:]
        for reader in out.get("readers") or []:
            for part in reader.get("chunks") or []:
                for key in ("first", "last"):  # word indices, ``last`` one past the part's last word
                    if isinstance(part.get(key), int):
                        part[key] = part[key] if part[key] <= first else max(first, part[key] - n)
    # The second reader's own text has no times: a part that overlaps the range loses its text whole.
    for reader in out.get("readers") or []:
        if reader.get("role") == "primary":
            continue
        parts = reader.get("chunks")
        hit = False
        if isinstance(parts, list) and parts:
            for part in parts:
                begins, ends = part.get("start_ms") or 0, part.get("end_ms")
                if begins < end_ms and (ends is None or ends > start_ms) and (part.get("text") or part.get("rejected_text")):
                    part.pop("rejected_text", None)
                    part.update(text=None, redacted=True)
                    hit = True
        elif reader.get("text") and inside:
            reader.update(text=None, redacted=True)
            hit = True
        counts["reader_parts"] += hit
    markers = out.get("markers")
    if isinstance(markers, list):
        keep = [m for m in markers if not (isinstance(m, dict) and m.get("start_ms") is not None and start_ms <= m["start_ms"] < end_ms)]
        counts["markers"] = len(markers) - len(keep)
        out["markers"] = keep
    out["redactions"] = sorted(record, key=lambda r: r["start_ms"])
    return (out if out != data else data), counts


# ---------------------------------------------------------------------------- the audio
def _silence_in_place(path: Path, start_ms: int, end_ms: int, listen: bool = False) -> dict:
    """Silence the range in the file itself: written beside it, read back against it, then swapped in."""
    tmp = path.with_name(f".{path.name}.redact.tmp")
    try:
        done = opus.silence(path, tmp, start_ms, end_ms)
        if not done["silenced"]:
            return {"silenced": 0, "already_silent": done["already_silent"]}
        checked = opus.check(path, tmp, start_ms, end_ms)
        heard = opus.decoded_check(path, tmp, start_ms, end_ms) if listen else {"decoded": False, "why": "not asked"}
        os.replace(tmp, path)
        return {"silenced": done["silenced"], "already_silent": done["already_silent"],
                "duration_diff_ms": abs(checked["duration_ms_before"] - checked["duration_ms_after"]),
                "packets_in_range": checked["packets_in_range"], "packets_outside_unchanged": checked["packets_outside_unchanged"],
                "bytes_before": checked["bytes_before"], "bytes_after": checked["bytes_after"], "heard": heard}
    finally:
        tmp.unlink(missing_ok=True)


def ensure_audio(store: Store, memo_id: str, path: Path) -> bool:
    """Before anything reads a recording's audio: every range on its record is silent in it. True when the
    file was changed (it had been fetched again, or an earlier run was cut short). Never raises."""
    try:
        ranges = store.redactions_for(memo_id)
        if not ranges or not path.is_file():
            return False
        fp = _fingerprint(path)
        if all(r.get("audio_fp") == fp for r in ranges):
            return False
        if not opus.is_opus(path):
            log.error("Memo %s has a range taken out, and its audio is not a file that can be silenced", memo_id)
            return False
        changed = False
        for r in ranges:
            changed |= bool(_silence_in_place(path, r["start_ms"], r["end_ms"])["silenced"])
        _stamp(store, memo_id, audio_fp=_fingerprint(path))
        if changed:
            log.warning("Memo %s: a range on its record was silenced again in the audio", memo_id)
        return changed
    except Exception:  # noqa: BLE001 - the caller is a worker run
        log.exception("Memo %s: the audio could not be checked against its record of ranges", memo_id)
        return False


def ensure_library(store: Store, state: dict, memos, audio_path) -> None:
    """At the start of a worker run: ``ensure_audio`` for every memo with a record. When silencing was the
    only change to a file whose transcript stands, the state follows the file, so it is not transcribed again."""
    record = store.read_redactions()
    for memo in memos:
        if memo.id not in record or not memo.local or not memo.filename:
            continue
        path, entry = audio_path(memo), state.get(memo.id) or {}
        try:
            before = _fingerprint(path)
        except OSError:
            continue
        if ensure_audio(store, memo.id, path) and entry.get("ok") and str(entry.get("fp") or "").split(":")[0] == before.split(":")[0]:
            entry["fp"] = _fingerprint(path)


def _stamp(store: Store, memo_id: str, start_ms: int | None = None, end_ms: int | None = None, **fields) -> None:
    memos = store.read_redactions()
    for r in memos.get(memo_id) or []:
        if start_ms is None or (r.get("start_ms") == start_ms and r.get("end_ms") == end_ms):
            r.update(fields)
    store.write_redactions(memos)


# ---------------------------------------------------------------------------- plan
def _lines(old: str, new: str) -> tuple[list[int], list[int]]:
    """Lines of ``old`` (from 0) that go whole, and lines that are replaced by a shorter or changed one. In a
    changed stretch, the new lines before the range's line stand for its first old lines and those after it
    for its last; what is between went."""
    a, b = old.split("\n"), new.split("\n")
    removed: list[int] = []
    cut: list[int] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("equal", "insert"):
            continue
        old_lines = [i for i in range(i1, i2) if a[i].strip()]
        new_lines = [x for x in b[j1:j2] if x.strip()]
        marks = [k for k, x in enumerate(new_lines) if x.startswith("[poistettu ")]
        if not marks:
            cut += old_lines[:len(new_lines)] if new_lines else []
            removed += old_lines[len(new_lines):]
            continue
        head, tail = marks[0], len(new_lines) - marks[-1] - 1
        cut += old_lines[:head] + (old_lines[len(old_lines) - tail:] if tail else [])
        removed += old_lines[head:len(old_lines) - tail]
    return sorted(set(removed) - set(cut)), sorted(set(cut))


def plan(cfg, store: Store, memo_id: str, from_ms: int, to_ms: int) -> dict:
    """What taking the range out would do, in numbers. Changes nothing."""
    rec = Recording(cfg, store, memo_id)
    data, words = rec.data, rec.data.get("words") or []
    audio = {"path": str(rec.audio) if rec.audio else None, "format": "ogg/opus" if rec.audio else None,
             "supported": rec.audio is not None, "problem": rec.audio_problem,
             "bytes": rec.audio.stat().st_size if rec.audio else None}
    duration_ms = int(round(float(data["memo"].get("duration_s") or 0) * 1000))
    known = next((r for r in store.redactions_for(rec.memo_id) if r["start_ms"] == from_ms and r["end_ms"] == to_ms), None)
    if known:  # a range already on record: begun or done
        left = len(_inside(words, from_ms, to_ms)) if words else 0
        marker = notation.redaction_line(known)
        done = known.get("state") == "done" and not left and marker in rec.md.read_text(encoding="utf-8").split("\n")
        return {"memo_id": rec.memo_id, "start_ms": from_ms, "end_ms": to_ms, "duration_ms": duration_ms, "whole": False,
                "already": done, "begun": True, "words": left, "words_total": len(words), "source_tokens": 0, "markers": 0,
                "reader_parts": 0, "edge": {"before": [], "after": []},
                "transcript": {"path": str(rec.md), "lines_removed": [], "lines_cut": []}, "audio": audio, "files": []}
    cut = snap(data, from_ms, to_ms)
    new, counts = without(cfg, data, cut["start_ms"], cut["end_ms"], dt.date.today().isoformat())
    old_md = rec.md.read_text(encoding="utf-8")
    removed, shortened = _lines(old_md, rec.render(new))
    files = [str(rec.md), str(rec.sidecar), str(store.state_path), str(store.redactions_path)]
    return {
        "memo_id": rec.memo_id, "start_ms": cut["start_ms"], "end_ms": cut["end_ms"], "duration_ms": duration_ms,
        "whole": cut["whole"], "already": False,
        "words": counts["words"], "words_total": len(words), "source_tokens": counts["source_tokens"],
        "markers": counts["markers"], "reader_parts": counts["reader_parts"],
        "edge": {"before": [w.get("text", "") for w in words[max(0, cut["first"] - EDGE_WORDS):cut["first"]]],
                 "after": [w.get("text", "") for w in words[cut["last"] + 1:cut["last"] + 1 + EDGE_WORDS]]},
        "transcript": {"path": str(rec.md), "lines_removed": removed, "lines_cut": shortened},
        "audio": audio, "files": files + ([str(rec.audio)] if rec.audio else []),
        **({"roam": "this memo was pushed to Roam: the block there is not touched"} if rec.entry.get("roam") else {}),
    }


# ---------------------------------------------------------------------------- apply
def _replace_markers(node, markers: list) -> int:
    n = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "markers" and isinstance(value, list):
                if value != markers:
                    node[key] = copy.deepcopy(markers)
                    n += 1
            else:
                n += _replace_markers(value, markers)
    elif isinstance(node, list):
        n += sum(_replace_markers(x, markers) for x in node)
    return n


def _patch_reports(store: Store, memo_id: str, start_ms: int, end_ms: int) -> int:
    """The second-reader reports list disagreements with the words around them: samples of this memo in or
    near the range go, from the JSON and from its Markdown twin."""
    lo, hi = start_ms - REPORT_PAD_MS, end_ms + REPORT_PAD_MS
    gone = 0
    for folder in {store.output_dir / "_reports", *(d / "_reports" for d in store.folder_dirs.values())}:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            report = read_json(path)
            n = 0
            for group in (report.get("languages") or {}).values() if isinstance(report, dict) else []:
                samples = group.get("samples") if isinstance(group, dict) else None
                if isinstance(samples, list):
                    keep = [s for s in samples if not (isinstance(s, dict) and s.get("memo_id") == memo_id
                                                       and isinstance(s.get("start_ms"), (int, float)) and lo <= s["start_ms"] < hi)]
                    n += len(samples) - len(keep)
                    group["samples"] = keep
            if n:
                write_json_atomic(path, report)
            md, m = path.with_suffix(".md"), 0
            if md.exists():
                out, ours, drop = [], False, False
                for line in md.read_text(encoding="utf-8").split("\n"):
                    if line.startswith("#"):
                        ours, drop = line.startswith("### ") and f"`{memo_id}`" in line, False
                    elif line.startswith("- [ ] at "):
                        at = re.match(r"- \[ \] at (?:(\d+):)?(\d+):(\d\d)", line)
                        ms = (int(at.group(1) or 0) * 3600 + int(at.group(2)) * 60 + int(at.group(3))) * 1000 if at else None
                        drop = bool(ours and ms is not None and lo <= ms < hi)
                        m += drop
                    elif not line.startswith("  "):
                        drop = False
                    if not drop:
                        out.append(line)
                if m:
                    write_text_atomic(md, "\n".join(out))
            gone += max(n, m)
    return gone


def apply(cfg, store: Store, memo_id: str, start_ms: int, end_ms: int, day: str) -> dict:
    """Take the range out of every form of the recording. The bounds are the ones ``plan`` gave."""
    try:
        dt.date.fromisoformat(str(day)[:10])
    except ValueError:
        raise RedactError("--date must be YYYY-MM-DD") from None
    with worker_lock(store):
        rec = Recording(cfg, store, memo_id)
        memos = store.read_redactions()
        mine = next((r for r in memos.get(rec.memo_id) or [] if r.get("start_ms") == start_ms and r.get("end_ms") == end_ms), None)
        if mine is None:
            cut = snap(rec.data, start_ms, end_ms)
            if (cut["start_ms"], cut["end_ms"]) != (start_ms, end_ms):
                raise RedactError("these are not the bounds plan gave for the recording as it is now; plan again")
            if cut["whole"]:
                raise RedactError("the range is the whole recording: remove the recording instead")
            if rec.audio is None:
                raise RedactError(f"the audio cannot be silenced ({rec.audio_problem}); nothing was changed")
            mine = {"start_ms": start_ms, "end_ms": end_ms, "date": str(day)[:10], "state": "begun"}
            memos.setdefault(rec.memo_id, []).append(mine)
            store.write_redactions(memos)  # first: from here on every writer and reader honours the range
        day = mine["date"]
        marker = notation.redaction_line(mine)
        already = mine.get("state") == "done"

        audio: dict = {"problem": rec.audio_problem}
        if rec.audio is not None:
            audio = _silence_in_place(rec.audio, start_ms, end_ms, listen=True)
            already &= not audio["silenced"]

        new, counts = without(cfg, rec.data, start_ms, end_ms, day)
        if new is not rec.data or marker not in (rec.md.read_text(encoding="utf-8").split("\n") if rec.md.exists() else []):
            result = store.rewrite_transcript(rec.entry["json"], new, cfg.threshold, cfg.run_min, cfg.pause_ms, cfg.marker_spec)
            rec.entry.update(stats=result["stats"], markers=result["markers"])
            already = False
        if marker not in rec.md.read_text(encoding="utf-8").split("\n"):
            raise RedactError("the transcript was written without the range's line; nothing more was done")

        markers = rec.entry.get("markers") if isinstance(rec.entry.get("markers"), list) else []
        if rec.audio is not None:
            rec.entry["fp"] = _fingerprint(rec.audio)
        if store.read_state() != rec.state:
            store.write_state(rec.state)
            already = False
        elsewhere = 0
        for path in (store.state_path.with_name(store.state_path.name + ".bak"), store.library_path):
            other = read_json(path)
            if isinstance(other, dict):
                own = other[rec.memo_id] if rec.memo_id in other else \
                    [m for m in (other.get("memos") or []) if isinstance(m, dict) and m.get("id") == rec.memo_id]
                n = _replace_markers(own, markers)
                if n:
                    write_json_atomic(path, other)
                    elsewhere += n
        cache = store.tmp_dir / rec.memo_id
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
        samples = _patch_reports(store, rec.memo_id, start_ms, end_ms)
        _stamp(store, rec.memo_id, start_ms, end_ms, state="done",
               **({"audio_fp": _fingerprint(rec.audio)} if rec.audio is not None else {}))
        return {"memo_id": rec.memo_id, "start_ms": start_ms, "end_ms": end_ms, "marker": marker, "already": bool(already),
                "counts": {**counts, "marker_lists_elsewhere": elsewhere, "report_samples": samples},
                "audio": audio, "files": [str(rec.md), str(rec.sidecar)] + ([str(rec.audio)] if rec.audio else []),
                **({"roam": "this memo was pushed to Roam: the block there is not touched"} if rec.entry.get("roam") else {})}


# ---------------------------------------------------------------------------- verify, list
def verify(cfg, store: Store, memo_id: str) -> dict:
    """Is every range on the memo's record still out of every form? Numbers only."""
    rec = Recording(cfg, store, memo_id)
    words, tokens = rec.data.get("words") or [], rec.data.get("source_tokens") or []
    lines = rec.md.read_text(encoding="utf-8").split("\n") if rec.md.exists() else []
    ranges = []
    for r in store.redactions_for(rec.memo_id):
        a, b = r["start_ms"], r["end_ms"]
        loud = opus.unsilent(rec.audio, a, b) if rec.audio is not None else None
        ranges.append({
            "start_ms": a, "end_ms": b, "date": r.get("date"), "state": r.get("state"),
            "words_in_range": sum(1 for w in words if w.get("start_ms") is not None and a <= w["start_ms"] < b),
            "tokens_in_range": sum(1 for t in tokens if isinstance(t, dict) and t.get("start_ms") is not None and a <= t["start_ms"] < b),
            "marker_in_transcript": notation.redaction_line(r) in lines,
            "on_the_transcripts_record": any(x.get("start_ms") == a and x.get("end_ms") == b for x in rec.data.get("redactions") or []),
            "audio_packets_not_silent": loud,
        })
    ok = bool(ranges) and all(r["state"] == "done" and not r["words_in_range"] and not r["tokens_in_range"] and r["marker_in_transcript"]
                              and r["on_the_transcripts_record"] and r["audio_packets_not_silent"] in (0, None) for r in ranges)
    return {"memo_id": rec.memo_id, "ok": ok, "audio": "checked" if rec.audio is not None else rec.audio_problem, "ranges": ranges}


def listing(store: Store) -> dict:
    return {"memos": {memo_id: [{k: r.get(k) for k in ("start_ms", "end_ms", "date", "state")} for r in ranges]
                      for memo_id, ranges in sorted(store.read_redactions().items())}}
