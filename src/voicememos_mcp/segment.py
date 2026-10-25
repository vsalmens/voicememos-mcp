"""Recordings longer than the primary engine accepts: cut, transcribe in parts, merge.

Soniox takes at most 300 minutes per file (see ``VOICEMEMOS_MAX_DURATION_MIN``). A longer
memo is cut into N parts of at most ``Config.segment_max_s`` each, N as small as possible.
Each cut is moved to the longest silence within ``VOICEMEMOS_SEGMENT_SILENCE_WINDOW_MIN`` of
the even split (ffmpeg ``silencedetect``), so a word is rarely cut in half. The parts are
copied without re-encoding (``-c copy``).

Merging shifts each part's times by the part's start, so the words, paragraph time stamps
and statistics run continuously from 0 to the end. Speaker labels from diarization only
hold within one engine run, and Soniox returns no speaker embeddings to match them across
parts, so a speaker key gets the part number in front (``"2:1"`` = speaker 1 of part 2) and
the label a letter for the part (``S1b``). The voiceprint is applied to every part.
"""

from __future__ import annotations

import bisect
import math
import re
import subprocess
from pathlib import Path

from . import notation
from .audio import cut_audio, find_tool

SILENCE_NOISE = "-35dB"
SILENCE_MIN_S = 1.0
_SILENCE = re.compile(r"silence_(start|end): (-?[\d.]+)")


class SegmentError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- planning

def part_count(duration_s: float, part_max_s: float) -> int:
    if part_max_s <= 0:
        raise SegmentError("VOICEMEMOS_SEGMENT_MARGIN_MIN leaves no room for a part")
    return max(1, math.ceil(duration_s / part_max_s))


def find_silences(audio: Path, start_s: float, end_s: float) -> list[tuple[float, float]]:
    """Silences (start, end) in seconds of the whole recording, within ``start_s``..``end_s``."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise SegmentError("ffmpeg is needed to split long recordings")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-ss", f"{max(start_s, 0):.3f}", "-t", f"{end_s - start_s:.3f}",
         "-i", str(audio), "-map", "0:a:0", "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_S}",
         "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise SegmentError(f"ffmpeg silencedetect failed: {result.stderr.strip()[-200:]}")
    silences, opened = [], None
    for kind, value in _SILENCE.findall(result.stderr):
        t = max(start_s, 0) + float(value)  # times are relative to the seek point
        if kind == "start":
            opened = t
        elif opened is not None:
            silences.append((opened, t))
            opened = None
    return silences


def plan_bounds(duration_s: float, part_max_s: float, window_s: float = 0.0,
                silences=find_silences, audio: Path | None = None) -> list[float]:
    """Part boundaries ``[0, cut1, ..., duration]`` with every part at most ``part_max_s``.

    With ``window_s`` each cut moves to the middle of the longest silence within that distance
    of the even split, as far as both neighbouring parts stay within the limit; without a
    silence it stays at the even split.
    """
    n = part_count(duration_s, part_max_s)
    bounds = [0.0]
    for k in range(1, n):
        target = k * duration_s / n
        lo = max(duration_s - (n - k) * part_max_s, bounds[-1])  # the rest must still fit
        hi = min(bounds[-1] + part_max_s, duration_s)  # this part must fit
        cut = min(max(target, lo), hi)
        if window_s > 0:
            a, b = max(target - window_s, lo), min(target + window_s, hi)
            found = [(e - s, -abs((s + e) / 2 - target), (s + e) / 2) for s, e in silences(audio, a, b)]
            found = [f for f in found if a <= f[2] <= b]
            if found:
                cut = max(found)[2]
        bounds.append(round(cut, 3))
    bounds.append(round(duration_s, 3))
    return bounds


# ----------------------------------------------------------------------------- audio

def probe(path: Path) -> tuple[str, float]:
    """(codec, duration in s) of the first audio stream."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise SegmentError("ffprobe (part of ffmpeg) is needed to check the parts")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name:format=duration",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode != 0 or not values.get("codec_name"):
        raise SegmentError(f"{path.name} is not a readable audio file: {result.stderr.strip()[:200]}")
    try:
        return values["codec_name"], float(values.get("duration", "nan"))
    except ValueError:
        return values["codec_name"], float("nan")


def cut_part(audio: Path, start_s: float, end_s: float | None, out: Path, tolerance_s: float = 2.0) -> Path:
    """Copy one part without re-encoding and check that it is a valid file of the expected length."""
    cut_audio(audio, start_s, end_s, out)
    codec, duration = probe(out)
    if end_s is not None and not abs(duration - (end_s - start_s)) <= tolerance_s:
        raise SegmentError(f"{out.name} is {duration:.1f} s long, expected {end_s - start_s:.1f} s")
    return out


# ----------------------------------------------------------------------------- merging

def _shift(item: dict, offset_ms: int, part: int) -> dict:
    out = dict(item)
    for key in ("start_ms", "end_ms"):
        if out.get(key) is not None:
            out[key] += offset_ms
    if out.get("speaker") not in (None, ""):
        out["speaker"] = f"{part}:{out['speaker']}"
    return out


def merge_parts(parts: list[dict]) -> tuple[list[dict], bool, list[dict], list[list[int]]]:
    """Merge per-part engine results into one recording.

    ``parts`` is ``[{"start_s": ..., "tokens": [...]}]`` in order. Words are built per part (the
    first token of a part must not join the last word of the previous one), then shifted.
    Returns ``(words, had_confidence, source_tokens, token_ranges)``; ``source_tokens`` are the
    engine's tokens with the same shift and speaker keys, and ``token_ranges`` say which of
    them came from which part, so each part's raw response can be recovered.
    """
    words, tokens, ranges, had_confidence = [], [], [], False
    for number, part in enumerate(parts, start=1):
        offset = int(round(part["start_s"] * 1000))
        part_words, confidence = notation.tokens_to_words(part["tokens"])
        had_confidence |= confidence
        words += [_shift(w, offset, number) for w in part_words]
        ranges.append([len(tokens), len(tokens) + len(part["tokens"])])
        tokens += [_shift(t, offset, number) for t in part["tokens"]]
    return words, had_confidence, tokens, ranges


def part_of(word: dict, bounds_ms: list[int]) -> int:
    """0-based part index of a word, from its start time."""
    start = word.get("start_ms") or 0
    return max(0, min(bisect.bisect_right(bounds_ms, start) - 1, len(bounds_ms) - 2))


def split_words(words: list[dict], bounds_s: list[float]) -> list[list[dict]]:
    bounds_ms = [int(round(b * 1000)) for b in bounds_s]
    out: list[list[dict]] = [[] for _ in range(len(bounds_s) - 1)]
    for word in words:
        out[part_of(word, bounds_ms)].append(word)
    return out


def part_letter(index: int) -> str:
    """0 -> a, 25 -> z, 26 -> aa."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("a") + rem) + letters
    return letters


def speaker_labels(words: list[dict]) -> dict:
    """Like ``notation.speaker_labels`` for merged words (keys ``"part:speaker"``).

    Speakers are numbered per part in order of appearance, with the part's letter: S1a, S2a,
    S1b, ... ``{}`` when no part has more than one speaker (the memo is then treated like a
    single-speaker recording, as each part would be on its own).
    """
    counts: dict[str, int] = {}
    order: dict[str, list[str]] = {}
    for word in words:
        spk = word.get("speaker")
        if spk in (None, ""):
            continue
        spk = str(spk)
        part = spk.split(":", 1)[0]
        if spk not in counts:
            order.setdefault(part, []).append(spk)
            counts[spk] = 0
        counts[spk] += 1
    if all(len(keys) < 2 for keys in order.values()):
        return {}
    total = sum(counts.values())
    out = {}
    for part, keys in order.items():
        letter = part_letter(int(part) - 1) if part.isdigit() else part
        for number, spk in enumerate(keys, start=1):
            out[spk] = {"label": f"S{number}{letter}", "words": counts[spk], "share": round(counts[spk] / total, 3)}
    return out


def clock(seconds: float) -> str:
    sec = int(seconds)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
