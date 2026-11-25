"""How the owner sounded in one recording, against how they usually sound (optional, fully local).

Measured, not guessed: nothing here names a feeling. With ``VOICEMEMOS_TONE=1`` each new recording
in which the voiceprint recognised the owner gets a few figures of the owner's own speech:

* from the engine's word times: speech rate (syllables per second of speaking) and the share of
  pauses inside the owner's own turns;
* from the audio, with Praat (the ``praat-parselmouth`` package): the median pitch and how much it
  moves (semitones), the level (dB), jitter and shimmer.

Only the owner's voice is measured. Who that is comes from the voiceprint alone: a firm match of a
speaker, or of a recording with one voice. A possible match, no voiceprint, or a recording without
the owner yields nothing at all. Other speakers are never measured.

The figures themselves say little: a microphone, a car, a cold all move them. So each is held
against the owner's own baseline, the median of the 30 recordings before this one (by recording
time) that have the figure, and what is written out is the deviation in robust units ((x - median)
/ (1.4826 * MAD)). The baseline is of recordings made earlier than this one, so the same recording
measured twice gives the same result whatever was processed in between.

Where it goes:

* ``<transcript>.tone.json`` beside the transcript's JSON: recording id, time, how many recordings
  the baseline had, the deviations and the words chosen for them. No raw figure, no word of the
  recording.
* one line in the transcript's header, ``[sävy: nopea, korkea, vähän taukoja]`` (the owner's
  language; ``[sävy: tavallinen]`` when nothing stands out). The transcript's text is not touched.
* ``DATA_DIR/tone.json`` (mode 600): the raw figures per recording, which the baseline is made of.

Until ``MIN_BASELINE`` earlier recordings exist there is no deviation and no line: the side file
says how many there are. The audio analysis looks at no more than ``MAX_AUDIO_S`` seconds of the
owner's speech, so the cost does not grow with the recording.

Needs ``pip install voicememos-mcp[tone]`` for the audio figures; without it the two figures from
the word times are still made and the side file says the rest is missing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Callable

from .store import now_iso, read_json, write_json_atomic, write_text_atomic

log = logging.getLogger(__name__)

SCHEMA = 1
SAMPLE_RATE = 16000
BASELINE_N = 30
MIN_BASELINE = 5
MAX_AUDIO_S = 120.0
MIN_SPEECH_S = 10.0       # less of the owner's speech than this is not measured
PAUSE_MIN_MS = 250        # a gap between two of the owner's words shorter than this is not a pause
PAUSE_MAX_MS = 10_000     # a longer gap is not a pause in speech (the recorder was left on)
STANDS_OUT = 1.0          # robust units from which a figure gets a word in the line

MEASURES = ("rate", "pause_share", "pitch", "pitch_var", "level", "jitter", "shimmer")
# The words of the line, in the owner's language: (above, below). Jitter and shimmer are in the
# side file only: there is no plain word for them that would not read as a judgement.
WORDS = {
    "rate": ("nopea", "hidas"),
    "pitch": ("korkea", "matala"),
    "pitch_var": ("vaihteleva", "tasainen"),
    "level": ("voimakas", "hiljainen"),
    "pause_share": ("paljon taukoja", "vähän taukoja"),
}
ORDER = ("rate", "pitch", "pitch_var", "level", "pause_share")
_VOWELS = re.compile(r"[aeiouyäöåáéíóúàèü]+", re.I)


def owner_speaker(speakers: dict, voice: dict | None) -> tuple[bool, str | None]:
    """(is the owner in the recording, their speaker key or None for a recording with one voice).
    Only a firm voiceprint match counts."""
    if not voice or voice.get("result") != "match":
        return False, None
    if speakers:
        key = voice.get("speaker")
        if key is None or (speakers.get(key) or {}).get("basis") != "voiceprint":
            return False, None
        return True, key
    return True, None


def _own(words: list[dict], speaker: str | None) -> list[dict]:
    return [w for w in words if (speaker is None or str(w.get("speaker")) == speaker)
            and w.get("start_ms") is not None and w.get("end_ms") is not None and (w.get("text") or "").strip()]


def syllables(text: str) -> int:
    """Vowel groups: close enough for Finnish and English to compare one speaker with themselves."""
    return max(1, len(_VOWELS.findall(text)))


def from_times(words: list[dict], speaker: str | None) -> dict:
    """Speech rate and pause share of one speaker from the word times. A pause is a gap between
    two consecutive words of the speaker's own turn (no word of anyone else in between)."""
    speaking_ms = pauses_ms = count = 0
    previous = None
    for word in words:
        if word.get("start_ms") is None or word.get("end_ms") is None or not (word.get("text") or "").strip():
            continue
        if speaker is not None and str(word.get("speaker")) != speaker:
            previous = None
            continue
        speaking_ms += max(0, word["end_ms"] - word["start_ms"])
        count += syllables(word["text"])
        if previous is not None:
            gap = word["start_ms"] - previous
            if PAUSE_MIN_MS <= gap <= PAUSE_MAX_MS:
                pauses_ms += gap
        previous = word["end_ms"]
    if speaking_ms < MIN_SPEECH_S * 1000:
        return {}
    return {"rate": round(count / (speaking_ms / 1000), 4),
            "pause_share": round(pauses_ms / (speaking_ms + pauses_ms), 4)}


def owner_spans(words: list[dict], speaker: str | None, limit_s: float = MAX_AUDIO_S) -> list[tuple[float, float]]:
    """The owner's longest uninterrupted stretches, in time order, up to ``limit_s`` in all."""
    from .voiceprint import speech_spans

    spans = sorted(speech_spans(words, speaker, min_s=1.0), key=lambda s: (-(s[1] - s[0]), s[0]))
    chosen, total = [], 0.0
    for a, b in spans:
        if total >= limit_s:
            break
        b = min(b, a + (limit_s - total))
        chosen.append((a, b))
        total += b - a
    return sorted(chosen)


def acoustic(samples, spans: list[tuple[float, float]]) -> dict:
    """Pitch, level, jitter and shimmer of the given stretches, with Praat. Raises ImportError
    when ``praat-parselmouth`` or numpy is not installed."""
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    from .voiceprint import as_reader

    read = as_reader(samples)
    parts = [np.asarray(read(a, b - a), dtype=np.float64) for a, b in spans]
    parts = [p for p in parts if len(p)]
    if not parts or sum(len(p) for p in parts) < MIN_SPEECH_S * SAMPLE_RATE / 2:
        return {}
    sound = parselmouth.Sound(np.concatenate(parts), sampling_frequency=SAMPLE_RATE)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    out: dict = {}
    if len(f0) >= 50:
        median = float(np.median(f0))
        semitones = 12 * np.log2(f0 / median)
        out["pitch"] = round(median, 2)
        out["pitch_var"] = round(float(np.percentile(semitones, 84) - np.percentile(semitones, 16)) / 2, 4)
    level = sound.get_intensity()
    if level is not None and math.isfinite(level):
        out["level"] = round(float(level), 2)
    points = call(sound, "To PointProcess (periodic, cc)", 60, 500)
    for name, value in (("jitter", call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)),
                        ("shimmer", call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))):
        if value is not None and math.isfinite(value):
            out[name] = round(float(value), 5)
    return out


# ----------------------------------------------------------------------------- baseline

def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def deviations(raw: dict, earlier: list[dict]) -> tuple[dict, int]:
    """Each figure of ``raw`` against the median of the same figure in the ``earlier`` recordings
    (already the last BASELINE_N), in robust units. (deviations, how many recordings the baseline
    had). A figure the baseline does not have enough of, or that does not vary there, is left out."""
    out: dict = {}
    for name in MEASURES:
        if name not in raw:
            continue
        values = [e["raw"][name] for e in earlier if name in e.get("raw", {})]
        if len(values) < MIN_BASELINE:
            continue
        median = _median(values)
        spread = 1.4826 * _median([abs(v - median) for v in values])
        if spread <= 0:
            continue
        out[name] = round((raw[name] - median) / spread, 2)
    return out, len(earlier)


def labels(dev: dict) -> list[str]:
    return [WORDS[name][0 if dev[name] > 0 else 1] for name in ORDER if name in dev and abs(dev[name]) >= STANDS_OUT]


def line(dev: dict) -> str | None:
    """``[sävy: …]`` or None when there is no deviation to speak of at all (no baseline yet)."""
    if not any(name in dev for name in ORDER):
        return None
    return f"[sävy: {', '.join(labels(dev)) or 'tavallinen'}]"


class History:
    """DATA_DIR/tone.json: the raw figures per recording (mode 600). No word of a recording."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        data = read_json(self.path, {})
        return data if isinstance(data, dict) and isinstance(data.get("memos"), dict) else {"schema": SCHEMA, "memos": {}}

    def earlier(self, memo_id: str, recorded: str | None) -> list[dict]:
        """The BASELINE_N recordings before ``recorded``, newest first, never this recording."""
        rows = [{"id": key, **value} for key, value in self.load()["memos"].items()
                if key != memo_id and value.get("recorded") and recorded and value["recorded"] < recorded]
        rows.sort(key=lambda r: (r["recorded"], r["id"]), reverse=True)
        return rows[:BASELINE_N]

    def put(self, memo_id: str, recorded: str | None, raw: dict) -> None:
        data = self.load()
        entry = {"recorded": recorded, "raw": raw}
        if data["memos"].get(memo_id) == entry:
            return
        data["memos"][memo_id] = entry
        data["updated"] = now_iso()
        write_json_atomic(self.path, data)
        os.chmod(self.path, 0o600)

    def drop(self, memo_id: str) -> None:
        data = self.load()
        if data["memos"].pop(memo_id, None) is not None:
            write_json_atomic(self.path, data)
            os.chmod(self.path, 0o600)


def side_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + ".tone.json")


def measure(words: list[dict], speakers: dict, voice: dict | None, samples_of: Callable[[], object],
            measurer: Callable | None = None) -> tuple[dict, list[str]]:
    """The owner's raw figures, or {} when the owner is not (firmly) in the recording or says too
    little. ``samples_of`` is called only when there is something to measure. (raw, what is
    missing and why)."""
    present, speaker = owner_speaker(speakers, voice)
    if not present:
        return {}, []
    raw = from_times(words, speaker)
    if not raw:
        return {}, []
    missing: list[str] = []
    try:
        raw.update((measurer or acoustic)(samples_of(), owner_spans(words, speaker)))
    except ImportError:
        missing.append("pitch, level, jitter and shimmer: praat-parselmouth is not installed (pip install voicememos-mcp[tone])")
    except Exception as exc:  # noqa: BLE001 - the figures from the word times still stand
        log.warning("Tone: the audio could not be measured: %s", exc)
        missing.append(f"pitch, level, jitter and shimmer: {str(exc)[:120]}")
    return raw, missing


def record(memo: dict, raw: dict, missing: list[str], history: History) -> dict:
    """Store the raw figures and build the side file's content for one recording."""
    earlier = history.earlier(memo["id"], memo.get("recorded"))
    history.put(memo["id"], memo.get("recorded"), raw)
    dev, n = deviations(raw, earlier)
    out = {"schema": SCHEMA, "memo_id": memo["id"], "recorded": memo.get("recorded"),
           "baseline": {"recordings": n, "of": BASELINE_N, "needed": MIN_BASELINE},
           "deviations": dev, "labels": labels(dev), "line": line(dev),
           "unit": "robust units against the owner's own earlier recordings: (x - median) / (1.4826 * MAD)"}
    if missing:
        out["missing"] = missing
    if out["line"] is None:
        out["note"] = (f"no baseline yet: {n} earlier recording(s) with the owner's voice measured, "
                       f"{MIN_BASELINE} needed" if n < MIN_BASELINE else "the baseline does not vary in any figure")
    return out


def write_side(json_path: Path, content: dict) -> Path:
    path = side_path(json_path)
    previous = read_json(path)
    if previous != content:
        write_text_atomic(path, json.dumps(content, ensure_ascii=False, indent=1) + "\n")
    return path
