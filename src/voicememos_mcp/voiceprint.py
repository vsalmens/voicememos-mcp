"""Recognise the memo owner's voice (optional, fully local).

A voiceprint is the average speaker embedding of a few memos in which only the owner
speaks. Speakers in other memos are compared with it by cosine similarity. Embeddings
come from a small ONNX speaker model run with sherpa-onnx; nothing leaves the machine. The
default is 3D-Speaker CAM++ trained on a large Chinese and English corpus ("common advanced",
28 MB). On a test set of Finnish and English memos it separated the owner from other speakers
clearly, while VoxCeleb-only models gave other Finnish speakers nearly the same scores as the
owner, so recording conditions mattered more to them than the voice.

Calibration: when enrolling, each enrollment memo is scored against a voiceprint built
from the others (leave-one-out). The lowest of those scores sets the "match" threshold,
so the threshold reflects how much the owner's own voice varies across devices and rooms.

Labels:

* ``Name``      score >= match threshold and clearly above any other speaker
* ``Name (?)``  score >= maybe threshold, or not clearly above another speaker
* otherwise the anonymous label (S1, S2, ...) is kept

Known other voices ("not me", e.g. a relative whose voice is close to the owner's) can be
added from one speaker of a memo. They are never shown by name; they only stop a speaker
from being labelled as the owner when that speaker is as close, or nearly as close, to a
known other voice.

The voiceprint is biometric data. It is stored only in DATA_DIR/voiceprint.json (mode 600)
and never in transcripts; MCP tools only report a summary.

Requires the optional dependencies ``sherpa-onnx`` and ``numpy`` (``pip install
voicememos-mcp[voiceprint]``) and ffmpeg or macOS ``afconvert`` for decoding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable

from .audio import find_tool
from .store import now_iso

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_S = 3.0
HOP_S = 1.5
MIN_SEGMENT_S = 1.0
MAX_WINDOWS_PER_SPEAKER = 12   # about 30 s of speech: constant cost per memo
MAX_WINDOWS_ENROLL = 40
CLEAN_TURN_S = 3.0             # known other voices: only turns at least this long (after trimming)
TURN_TRIM_S = 0.4              # ... with this much cut from both ends
MARGIN = 0.05                  # owner must beat the next speaker by this much for a firm label
MAYBE_GAP = 0.08               # maybe threshold = match threshold - this
DEFAULT_BOUNDS = (0.65, 0.95)         # clamp for the match threshold when a model has no own bounds

MAX_MODEL_BYTES = 200 * 1024 * 1024

# The release tag "speaker-recongition-models" is spelled that way upstream; do not "fix" it.
MODELS = {
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
               "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "sha256": "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
        # Scores spread wider with this model (owner about 0.5-0.8, others below 0.4), so the
        # clamp, the "possible" band and the spread warning are scaled to it.
        "bounds": (0.35, 0.85), "maybe_gap": 0.05, "spread_warning": 0.25,
    },
    "wespeaker_en_voxceleb_resnet34_LM": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
               "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "sha256": "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
        "bounds": (0.65, 0.95),
    },
}
DEFAULT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced"


class VoiceprintError(RuntimeError):
    pass


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise VoiceprintError("numpy is required (pip install 'voicememos-mcp[voiceprint]')") from exc
    return np


# ----------------------------------------------------------------------------- audio

ENERGY_SCAN_S = 600  # without word timings, look at the first 10 minutes only


class FileReader:
    """Decode only the requested parts of an audio file (ffmpeg seeks), so long memos stay cheap."""

    def __init__(self, path: Path, ffmpeg: str):
        self.path, self.ffmpeg = path, ffmpeg

    def __call__(self, start_s: float, duration_s: float):
        np = _np()
        result = subprocess.run(
            [self.ffmpeg, "-nostdin", "-loglevel", "error", "-ss", f"{max(start_s, 0):.3f}", "-t",
             f"{duration_s:.3f}", "-i", str(self.path), "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "f32le", "-"],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise VoiceprintError(f"cannot decode {self.path.name}: {result.stderr.decode()[:200]}")
        return np.frombuffer(result.stdout, dtype=np.float32)


def load_audio(path: Path):
    """Return a reader for 16 kHz mono float32 audio: ``reader(start_s, duration_s) -> samples``."""
    ffmpeg = find_tool("ffmpeg")
    if ffmpeg:
        return FileReader(path, ffmpeg)
    np = _np()
    afconvert = shutil.which("afconvert")
    if afconvert:  # no seeking without ffmpeg: decode once
        with tempfile.TemporaryDirectory(prefix="voicememos-vp-") as tmp:
            out = Path(tmp) / "audio.wav"
            result = subprocess.run([afconvert, "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
                                     str(path), str(out)], capture_output=True, check=False)
            if result.returncode == 0 and out.exists():
                with wave.open(str(out)) as wav:
                    raw = wav.readframes(wav.getnframes())
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    raise VoiceprintError(f"cannot decode {path.name}: neither ffmpeg nor afconvert worked")


def as_reader(audio):
    """Accept a reader or an in-memory sample array."""
    if callable(audio):
        return audio

    def read(start_s: float, duration_s: float):
        a = int(max(start_s, 0) * SAMPLE_RATE)
        return audio[a: a + int(duration_s * SAMPLE_RATE)]

    return read


def speech_spans(words: list[dict], speaker: str | None = None, max_gap_ms: int = 500,
                 min_s: float = MIN_SEGMENT_S, trim_s: float = 0.0) -> list[tuple[float, float]]:
    """Time spans (s) where one speaker talks, built from word timings.

    A word by any other speaker ends the span, so a span never bridges someone else's
    interjection. ``trim_s`` cuts that much from both ends of every span (turn edges are where
    diarization most often errs); spans shorter than ``min_s`` after trimming are dropped.
    """
    spans: list[list[float]] = []
    open_span = False
    for word in words:
        if speaker is not None and str(word.get("speaker")) != speaker:
            open_span = False
            continue
        start, end = word.get("start_ms"), word.get("end_ms")
        if start is None or end is None:
            continue
        if open_span and spans and start - spans[-1][1] * 1000 <= max_gap_ms:
            spans[-1][1] = end / 1000
        else:
            spans.append([start / 1000, end / 1000])
        open_span = True
    trimmed = [(a + trim_s, b - trim_s) for a, b in spans]
    return [(a, b) for a, b in trimmed if b - a >= min_s]


def window_positions(spans: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    """(start, duration) of 3 s windows (or whole short segments) in the spans, evenly sampled up to ``limit``."""
    out = []
    for a, b in spans:
        if b - a <= WINDOW_S:
            if b - a >= MIN_SEGMENT_S:
                out.append((a, b - a))
            continue
        t = a
        while t + WINDOW_S <= b + 1e-6:
            out.append((t, WINDOW_S))
            t += HOP_S
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def windows_from_spans(audio, spans: list[tuple[float, float]], limit: int) -> list:
    reader = as_reader(audio)
    windows = [reader(start, dur) for start, dur in window_positions(spans, limit)]
    return [w for w in windows if len(w) >= int(MIN_SEGMENT_S * SAMPLE_RATE)]


def energy_windows(audio, limit: int) -> list:
    """Without word timings: the louder 60 % of 3 s windows (crude speech selection)."""
    np = _np()
    samples = as_reader(audio)(0, ENERGY_SCAN_S)
    win, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    segs = [samples[i:i + win] for i in range(0, max(1, len(samples) - win + 1), hop)]
    segs = [s for s in segs if len(s) >= int(MIN_SEGMENT_S * SAMPLE_RATE)]
    if not segs:
        return []
    energy = np.array([float(np.sqrt(np.mean(np.square(s)) + 1e-12)) for s in segs])
    keep = [s for s, e in zip(segs, energy) if e >= np.percentile(energy, 40)]
    if len(keep) > limit:
        step = len(keep) / limit
        keep = [keep[int(i * step)] for i in range(limit)]
    return keep


# ----------------------------------------------------------------------------- model

def ensure_model(models_dir: Path, name: str) -> Path:
    """Download the ONNX model once and verify its checksum."""
    if name not in MODELS:
        raise VoiceprintError(f"unknown voiceprint model '{name}'")
    spec = MODELS[name]
    path = models_dir / f"{name}.onnx"
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    import requests

    log.info("downloading voiceprint model %s", name)
    fd, tmp_name = tempfile.mkstemp(dir=models_dir, prefix=f".{name}.", suffix=".download")
    tmp = Path(tmp_name)
    try:
        written = 0
        with os.fdopen(fd, "wb") as fh, requests.get(spec["url"], stream=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                written += len(chunk)
                if written > MAX_MODEL_BYTES:
                    raise VoiceprintError("voiceprint model download is unexpectedly large; aborted")
                fh.write(chunk)
        if _sha256(tmp) != spec["sha256"]:
            raise VoiceprintError("voiceprint model checksum mismatch; download aborted")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SherpaEmbedder:
    """Speaker embeddings with sherpa-onnx."""

    def __init__(self, model_path: Path, threads: int = 2):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise VoiceprintError("sherpa-onnx is required (pip install 'voicememos-mcp[voiceprint]')") from exc
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=threads)
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def __call__(self, samples):
        np = _np()
        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        return np.array(self.extractor.compute(stream), dtype=np.float32)


Embedder = Callable[[object], object]


def _unit(v):
    np = _np()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def mean_embedding(embed: Embedder, windows: list):
    np = _np()
    if not windows:
        return None
    return _unit(np.mean([_unit(embed(w)) for w in windows], axis=0))


def other_voice(name: str, samples, words: list[dict] | None, speaker: str | None, embed: Embedder) -> dict:
    """Embedding of a known other person ("not me"), from one speaker of one memo.

    Only longer uninterrupted turns are used, with their edges trimmed, so that words the
    diarization gave to the wrong speaker (short replies, turn changes) do not leak the
    owner's voice into the other person's vector.
    """
    spans = speech_spans(words, speaker, min_s=CLEAN_TURN_S, trim_s=TURN_TRIM_S) if words else []
    if words and len(window_positions(spans, MAX_WINDOWS_ENROLL)) < 3:
        spans = speech_spans(words, speaker)  # too little long speech: use every turn
    windows = windows_from_spans(samples, spans, MAX_WINDOWS_ENROLL) if spans else []
    if len(windows) < 3 and not speaker:
        windows = energy_windows(samples, MAX_WINDOWS_ENROLL)
    if len(windows) < 3:
        raise VoiceprintError("too little speech from that speaker")
    vector = mean_embedding(embed, windows)
    return {"name": name, "vector": [round(float(x), 6) for x in vector], "windows": len(windows)}


def _other_centroids(voiceprint: dict) -> dict:
    np = _np()
    groups: dict[str, list] = {}
    for other in voiceprint.get("others", []):
        groups.setdefault(other["name"], []).append(np.array(other["vector"], dtype=np.float32))
    return {name: _unit(np.mean(vectors, axis=0)) for name, vectors in groups.items()}


def _closest_other(vector, others: dict) -> tuple[str | None, float | None]:
    if not others:
        return None, None
    name = max(others, key=lambda n: float(vector @ others[n]))
    return name, round(float(vector @ others[name]), 4)


# ----------------------------------------------------------------------------- voiceprint file

class Voiceprint:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".voiceprint.", suffix=".tmp")  # mode 0600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        os.chmod(self.path, 0o600)

    def summary(self) -> dict:
        data = self.load()
        if not data:
            return {"enrolled": False}
        return {"enrolled": True, "name": data.get("name"), "model": data.get("model"),
                "created": data.get("created"), "memos": [e["memo_id"] for e in data.get("enrolled", [])],
                "own_scores": data.get("own_scores"), "thresholds": data.get("thresholds"),
                "warnings": data.get("warnings", []),
                "other_voices": sorted({o["name"] for o in data.get("others", [])})}


# ----------------------------------------------------------------------------- enroll / identify

def enroll(name: str, model: str, samples_by_memo: dict[str, object], embed: Embedder,
           words_by_memo: dict[str, list[dict]] | None = None) -> dict:
    """Build a voiceprint from memos where only the owner speaks, with leave-one-out calibration."""
    np = _np()
    if len(samples_by_memo) < 2:
        raise VoiceprintError("enroll at least 2 memos (4-6 from different years and places is better)")
    words_by_memo = words_by_memo or {}
    per_memo = {}
    enrolled = []
    for memo_id, samples in samples_by_memo.items():
        words = words_by_memo.get(memo_id)
        windows = windows_from_spans(samples, speech_spans(words), MAX_WINDOWS_ENROLL) if words else []
        if len(windows) < 3:  # no usable word timings: fall back to the louder parts of the audio
            windows = energy_windows(samples, MAX_WINDOWS_ENROLL)
        if len(windows) < 3:
            raise VoiceprintError(f"memo {memo_id[:8]} has too little speech for enrollment")
        vectors = np.array([_unit(embed(w)) for w in windows])
        per_memo[memo_id] = vectors
        enrolled.append({"memo_id": memo_id, "windows": len(windows)})

    # Every memo weighs the same, so one long memo cannot dominate the voiceprint.
    memo_means = {k: _unit(v.mean(axis=0)) for k, v in per_memo.items()}
    centroid = _unit(np.mean(list(memo_means.values()), axis=0))
    own_scores = {}
    for memo_id, vectors in per_memo.items():
        reference = _unit(np.mean([m for k, m in memo_means.items() if k != memo_id], axis=0))
        sample = vectors[: MAX_WINDOWS_PER_SPEAKER]  # score like a speaker in a normal memo
        own_scores[memo_id] = round(float(_unit(sample.mean(axis=0)) @ reference), 4)

    lowest = min(own_scores.values())
    spec = MODELS.get(model, {})
    floor, ceil = spec.get("bounds", DEFAULT_BOUNDS)
    match = float(min(max(lowest - 0.02, floor), ceil))
    maybe_gap = spec.get("maybe_gap", MAYBE_GAP)
    warnings = []
    if len(samples_by_memo) < 4:
        warnings.append("fewer than 4 enrollment memos: thresholds are rough")
    spread = max(own_scores.values()) - lowest
    if spread > spec.get("spread_warning", 0.15):
        warnings.append("enrollment memos differ a lot from each other; check that only you speak in them")
    return {
        "name": name,
        "model": model,
        "created": now_iso(),
        "dim": int(centroid.shape[0]),
        "centroid": [round(float(x), 6) for x in centroid],
        "enrolled": enrolled,
        "own_scores": own_scores,
        "thresholds": {"match": round(match, 4), "maybe": round(match - maybe_gap, 4), "margin": MARGIN},
        "warnings": warnings,
    }


def identify(samples, words: list[dict], speakers: dict, voiceprint: dict, embed: Embedder) -> tuple[dict, dict]:
    """Score each speaker against the voiceprint and relabel the owner.

    ``speakers`` comes from ``notation.speaker_labels`` ({} for single-speaker memos).
    Returns (updated speakers, summary for the transcript JSON).
    """
    np = _np()
    centroid = _unit(np.array(voiceprint["centroid"], dtype=np.float32))
    thresholds = voiceprint["thresholds"]
    name = voiceprint["name"]
    others = _other_centroids(voiceprint)

    if not speakers:  # one speaker (or no diarization): score the whole memo
        windows = windows_from_spans(samples, speech_spans(words), MAX_WINDOWS_PER_SPEAKER)
        if len(windows) < 2:
            windows = energy_windows(samples, MAX_WINDOWS_PER_SPEAKER)
        vector = mean_embedding(embed, windows)
        if vector is None:
            return speakers, {"name": name, "score": None, "result": "too little speech"}
        score = round(float(vector @ centroid), 4)
        result = ("match" if score >= thresholds["match"] else "possible" if score >= thresholds["maybe"]
                  else "no match")
        summary = {"name": name, "score": score, "result": result, "thresholds": thresholds}
        other, other_score = _closest_other(vector, others)
        if other is not None:
            summary["closest_other"] = {"name": other, "score": other_score}
            if other_score >= score:
                summary["result"] = "no match"
            elif result == "match" and score - other_score < thresholds.get("margin", MARGIN):
                summary["result"] = "possible"
        return speakers, summary

    scores, closest = {}, {}
    for spk in speakers:
        vector = mean_embedding(embed, windows_from_spans(samples, speech_spans(words, spk),
                                                          MAX_WINDOWS_PER_SPEAKER))
        if vector is not None:
            scores[spk] = round(float(vector @ centroid), 4)
            other, other_score = _closest_other(vector, others)
            if other is not None:
                closest[spk] = {"name": other, "score": other_score}
    updated = {k: dict(v) for k, v in speakers.items()}
    for spk, score in scores.items():
        updated[spk]["voiceprint"] = score
    summary = {"name": name, "scores": scores, "thresholds": thresholds, "result": "no match"}
    if not scores:
        summary["result"] = "too little speech"
        return updated, summary
    if closest:
        summary["closest_other"] = closest
    best = max(scores, key=scores.get)
    rest = [s for k, s in scores.items() if k != best]
    margin = scores[best] - max(rest) if rest else 1.0
    if best in closest:  # a known other voice must also be clearly further away than the owner's
        if closest[best]["score"] >= scores[best]:
            return updated, summary
        margin = min(margin, scores[best] - closest[best]["score"])
    if scores[best] >= thresholds["match"] and margin >= thresholds.get("margin", MARGIN):
        updated[best].update(label=name, basis="voiceprint")
        summary.update(result="match", speaker=best)
    elif scores[best] >= thresholds["maybe"]:
        updated[best].update(label=f"{name} (?)", basis="voiceprint")
        summary.update(result="possible", speaker=best)
    return updated, summary


def suggest_enrollment(memos: list[dict], limit: int = 8) -> list[dict]:
    """Memos transcribed with diarization that found one speaker, 30 s to 15 min, spread over the years."""
    candidates = [m for m in memos
                  if (m.get("transcript") or {}).get("json") and (m.get("transcript") or {}).get("speakers") == 0
                  and m.get("local") and not m.get("deleted") and 30 <= (m.get("duration_s") or 0) <= 900
                  and m.get("recorded")]
    by_year: dict[str, list[dict]] = {}
    for memo in sorted(candidates, key=lambda m: -(m.get("duration_s") or 0)):
        by_year.setdefault(memo["recorded"][:4], []).append(memo)
    picked: list[dict] = []
    while len(picked) < limit and any(by_year.values()):
        for year in sorted(by_year):
            if by_year[year] and len(picked) < limit:
                picked.append(by_year[year].pop(0))
    return sorted(picked, key=lambda m: m["recorded"])
