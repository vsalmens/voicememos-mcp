"""How the words were said: measurements per word, fully local.

For every word with a time span three things are measured from the audio and stored relative to
the same speaker's own level in that recording:

* ``db``    loudness: the word's level minus the median level of the speaker's words (dB)
* ``st``    pitch: the word's median fundamental frequency against the speaker's median (semitones)
* ``rate``  tempo: the word's syllables per second divided by the speaker's median (1.0 = usual)

They are measurements, not a reading of feeling: a raised voice, a pitch jump or a stretched
word is something a reader can weigh, and nothing here says "angry" or "glad". The pause before
a word needs no measuring; it is in the word times already.

The speaker's own level is one figure per recording (``data["prosody"]["speakers"]``), so the
relative values can be turned back into dBFS, Hz and syllables per second. With diarization the
level is per speaker; without it the whole recording is one speaker.

Pitch is found per 40 ms frame by autocorrelation (Boersma 1993: the frame's autocorrelation
divided by the window's own; the best peak, with a small cost per octave of lag so that of two
equal peaks the higher pitch wins). A frame counts as voiced when that peak is at least
``VOICED_MIN`` and the frame is not silence. A word's pitch is the median over its voiced
frames. A word that still comes out about an octave from the speaker's own level is folded by
an octave when that brings it nearer: on speech through a codec nearly all such values are the
tracker's octave errors or creaky voice, not jumps a listener hears, so ``st`` stays within
about nine semitones either way.

Tempo is counted from the start of a word to the start of the next one when they follow each
other without a pause (the engine's word ends are coarse, its starts are not), else over the
word's own span. It is still rough for a single word; the view marks tempo for runs of words.

Needs numpy (``pip install voicememos-mcp[voiceprint]``) and ffmpeg or ``afconvert`` for decoding.
"""

from __future__ import annotations

import math
import re

SAMPLE_RATE = 16000
FRAME_S = 0.04
HOP_S = 0.01
F0_MIN_HZ = 60.0
F0_MAX_HZ = 420.0
VOICED_MIN = 0.5          # normalised autocorrelation peak of a voiced frame
SILENCE_DBFS = -60.0      # frames below this are not looked at for pitch
MIN_VOICED_FRAMES = 3     # a word needs this many voiced frames to get a pitch
MIN_WORD_MS = 60          # shorter spans are timing noise: no tempo
OCTAVE_COST = 0.02        # per octave of lag, off the peak's strength
FOLD_ST = 9.0             # a word this far from the speaker's level is folded by an octave if that is nearer
NEXT_WORD_MS = 250        # a next word starting within this of the word's end is "without a pause"
CHUNK_S = 300.0           # the audio is decoded in pieces of this length
VERSION = 1

VOWELS = "aeiouyäöåáàâãéèêëíìîïóòôõúùûüæø"
_VOWEL_GROUP = re.compile(f"[{VOWELS}]+", re.IGNORECASE)
_LETTERS = re.compile(r"[^\W\d_]+")


def syllables(text: str) -> int:
    """A rough syllable count: groups of vowels, a long group (Finnish "aio", "auo") as two.
    Good enough to compare a speaker's words with each other, which is all it is used for."""
    count = 0
    for letters in _LETTERS.findall(text or ""):
        groups = _VOWEL_GROUP.findall(letters)
        count += sum(2 if len(g) >= 3 else 1 for g in groups) or 1
    return count


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("numpy is required (pip install 'voicememos-mcp[voiceprint]')") from exc
    return np


def frame_features(samples):
    """Level (dBFS) and pitch (Hz, NaN when unvoiced) of each 40 ms frame, one every 10 ms.
    Returns ``(level, f0)``; frame ``k`` is centred at ``k * HOP_S + FRAME_S / 2``."""
    np = _np()
    win, hop = int(FRAME_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) < win:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, win)[::hop]
    n = len(frames)
    level = np.empty(n, dtype=np.float32)
    f0 = np.full(n, np.nan, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)
    size = 1 << (2 * win - 1).bit_length()
    window_acf = np.fft.irfft(np.abs(np.fft.rfft(window, size)) ** 2)[:win]
    lo = int(SAMPLE_RATE / F0_MAX_HZ)
    hi = min(int(SAMPLE_RATE / F0_MIN_HZ), win // 2)
    window_acf = window_acf[: hi + 2] / window_acf[0]
    cost = (OCTAVE_COST * np.log2(np.arange(lo, hi + 1) / lo)).astype(np.float32)
    for a in range(0, n, 4000):
        block = frames[a: a + 4000].astype(np.float32)
        power = np.mean(block * block, axis=1)
        level[a: a + len(block)] = 10 * np.log10(power + 1e-12)
        x = (block - block.mean(axis=1, keepdims=True)) * window
        acf = np.fft.irfft(np.abs(np.fft.rfft(x, size, axis=1)) ** 2, axis=1)[:, : hi + 2]
        zero = acf[:, :1]
        r = np.where(zero > 1e-10, acf / np.maximum(zero, 1e-10), 0.0) / window_acf
        body = r[:, lo: hi + 1]
        peak = (body > r[:, lo - 1: hi]) & (body >= r[:, lo + 1: hi + 2])
        scored = np.where(peak, body - cost, -9.0)
        pick = scored.argmax(axis=1)
        best = np.take_along_axis(body, pick[:, None], axis=1)[:, 0]
        lag = lo + pick
        voiced = (best >= VOICED_MIN) & (level[a: a + len(block)] > SILENCE_DBFS) & peak.any(axis=1)
        f0[a: a + len(block)] = np.where(voiced, SAMPLE_RATE / np.maximum(lag, 1), np.nan)
    return level, f0


def _word_values(read, words: list[dict]) -> list[tuple[float | None, float | None]]:
    """(level dBFS, pitch Hz) per word, decoding the audio piece by piece."""
    np = _np()
    out: list[tuple[float | None, float | None]] = [(None, None)] * len(words)
    timed = [(k, w["start_ms"] / 1000, w["end_ms"] / 1000) for k, w in enumerate(words)
             if w.get("start_ms") is not None and w.get("end_ms") is not None and w["end_ms"] > w["start_ms"]]
    timed.sort(key=lambda t: t[1])
    i = 0
    while i < len(timed):
        start = timed[i][1]
        j = i
        while j < len(timed) and timed[j][2] <= start + CHUNK_S:
            j += 1
        if j == i:  # one "word" longer than a piece (an engine's loop over music): measured alone
            j = i + 1
        end = max(t[2] for t in timed[i:j])
        origin = max(start - FRAME_S, 0.0)
        level, f0 = frame_features(read(origin, end - origin + FRAME_S))
        centres = origin + np.arange(len(level)) * HOP_S + FRAME_S / 2
        for k, a, b in timed[i:j]:
            lo, hi = np.searchsorted(centres, a), np.searchsorted(centres, b)
            if hi <= lo:
                lo, hi = max(lo - 1, 0), min(lo + 1, len(level))
            if hi <= lo:
                continue
            db = float(10 * np.log10(np.mean(10 ** (level[lo:hi] / 10)) + 1e-12))
            pitches = f0[lo:hi]
            pitches = pitches[~np.isnan(pitches)]
            out[k] = (db, float(np.median(pitches)) if len(pitches) >= MIN_VOICED_FRAMES else None)
        i = j
    return out


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def measure(audio, words: list[dict]) -> tuple[list[dict], dict]:
    """Return the words with ``pros`` added, and the summary for ``data["prosody"]``.

    ``audio`` is a reader ``(start_s, duration_s) -> samples`` at 16 kHz mono (see
    ``voiceprint.load_audio``) or the samples themselves. Words without a time span, and words
    whose speaker has too little speech for a level of their own, get no ``pros``."""
    from .voiceprint import as_reader

    raw = _word_values(as_reader(audio), words)
    rates: list[float | None] = []
    for k, word in enumerate(words):
        start, end = word.get("start_ms"), word.get("end_ms")
        n = syllables(word.get("text", ""))
        if start is not None and end is not None and k + 1 < len(words):
            following = words[k + 1]
            onset = following.get("start_ms")
            if onset is not None and following.get("speaker") == word.get("speaker") and end <= onset <= end + NEXT_WORD_MS:
                end = onset
        ok = start is not None and end is not None and end - start >= MIN_WORD_MS and n
        rates.append(n / ((end - start) / 1000) if ok else None)
    by_speaker: dict[str, list[int]] = {}
    for k, word in enumerate(words):
        by_speaker.setdefault(str(word.get("speaker") or ""), []).append(k)
    speakers: dict[str, dict] = {}
    out = [{key: value for key, value in w.items() if key != "pros"} for w in words]
    for speaker, indexes in by_speaker.items():
        base_db = _median([raw[k][0] for k in indexes if raw[k][0] is not None])
        base_f0 = _median([raw[k][1] for k in indexes if raw[k][1] is not None])
        base_rate = _median([rates[k] for k in indexes if rates[k] is not None])
        if base_db is None:
            continue
        speakers[speaker] = {"words": len(indexes), "level_dbfs": round(base_db, 1),
                             "f0_hz": round(base_f0, 1) if base_f0 else None,
                             "syllables_per_s": round(base_rate, 2) if base_rate else None}
        for k in indexes:
            db, f0 = raw[k]
            pros = {}
            if db is not None:
                pros["db"] = round(db - base_db, 1)
            if f0 is not None and base_f0:
                st = 12 * math.log2(f0 / base_f0)
                if abs(st) >= FOLD_ST and abs(st - math.copysign(12, st)) < abs(st):
                    st -= math.copysign(12, st)
                pros["st"] = round(st, 1)
            if rates[k] is not None and base_rate:
                pros["rate"] = round(rates[k] / base_rate, 2)
            if pros:
                out[k]["pros"] = pros
    summary = {"version": VERSION, "speakers": speakers,
               "measured_words": sum(1 for w in out if w.get("pros"))}
    return out, summary
