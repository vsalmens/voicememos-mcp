"""Audio preparation.

Soniox accepts common formats such as m4a, but not Apple's ``.qta`` (QuickTime
audio, used by Voice Memos for new recordings). A ``.qta`` file contains an AAC
track plus extra tracks, so the AAC track is copied into an ``.m4a`` container
without re-encoding (ffmpeg). Without ffmpeg, macOS's built-in ``afconvert`` is
used to write lossless ALAC instead. No denoising or trimming is done: both tend
to hurt recognition more than they help.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DIRECT_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".mp4", ".webm", ".aiff", ".amr"}
CONVERT_EXTENSIONS = {".qta", ".caf", ".mov"}


def find_tool(name: str) -> str | None:
    for directory in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidate = Path(directory) / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def prepare_audio(path: Path, workdir: Path) -> Path:
    """Return a path Soniox can read: the file itself, or a converted copy in ``workdir``."""
    suffix = path.suffix.lower()
    if suffix in DIRECT_EXTENSIONS:
        return path
    if suffix not in CONVERT_EXTENSIONS:
        raise RuntimeError(f"Unsupported audio format: {path.name}")

    out = workdir / (path.stem + ".m4a")
    errors: list[str] = []

    ffmpeg = find_tool("ffmpeg")
    if ffmpeg:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
             "-vn", "-map", "0:a:0", "-c:a", "copy", str(out)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        errors.append(f"ffmpeg: {result.stderr.strip()[:200]}")

    afconvert = shutil.which("afconvert")
    if afconvert:
        result = subprocess.run(
            [afconvert, "-f", "m4af", "-d", "alac", str(path), str(out)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        errors.append(f"afconvert: {result.stderr.strip()[:200]}")

    detail = "; ".join(errors) or "neither ffmpeg nor afconvert is available"
    raise RuntimeError(f"Could not convert {path.name} to m4a ({detail})")


def ogg_streams(path: Path) -> int:
    """How many streams an Ogg file starts with (each begins with a page of its own, flagged as
    the beginning of a stream, before any data page). 0 when the file is not Ogg."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(65536)
    except OSError:
        return 0
    serials = set()
    at = head.find(b"OggS")
    while at != -1 and at + 27 <= len(head):
        if head[at + 5] & 0x02:
            serials.add(head[at + 14: at + 18])
        at = head.find(b"OggS", at + 4)
    return len(serials)


def sound_only(path: Path, workdir: Path) -> Path:
    """The file itself, or for an Ogg file with a second stream beside the sound (a Plaud
    recorder writes one of zeros) a copy in ``workdir`` with the sound stream alone. Gemini
    refuses some such files whole ("the file or media data is empty, corrupted, or malformed":
    seen 26.11.2026 on an 18-second recording, which it read once the other stream was left out).
    The sound is copied, not encoded again. Without ffmpeg, or if the copy fails, the file itself."""
    if path.suffix.lower() not in (".ogg", ".opus", ".oga") or ogg_streams(path) < 2:
        return path
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        return path
    out = workdir / (path.stem + ".sound" + path.suffix)
    result = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                             "-vn", "-map", "0:a:0", "-c:a", "copy", str(out)],
                            capture_output=True, text=True, check=False)
    return out if out.exists() and out.stat().st_size > 0 else path


def cut_audio(path: Path, start_s: float, end_s: float | None, out: Path) -> Path:
    """Copy the part ``start_s``..``end_s`` (None = to the end) of an audio file without re-encoding."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is needed to split long recordings for the second reader")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(path)]
    if end_s is not None:
        cmd += ["-t", f"{max(end_s - start_s, 0.1):.3f}"]
    cmd += ["-vn", "-map", "0:a:0", "-c:a", "copy", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not cut the recording: {result.stderr.strip()[:200]}")
    return out
