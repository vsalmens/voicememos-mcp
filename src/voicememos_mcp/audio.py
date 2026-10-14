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
