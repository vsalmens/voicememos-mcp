"""Configuration.

Settings come from three places, later ones overriding earlier ones:

1. built-in defaults (below),
2. a private config file, ``~/.config/voicememos-mcp/config.env`` by default
   (override the path with ``VOICEMEMOS_CONFIG``),
3. environment variables.

Secrets (``SONIOX_API_KEY``, ``GEMINI_API_KEY``, ``ROAM_API_TOKEN``) are read from the environment or,
on macOS, from the login Keychain (service ``voicememos-mcp``, account = variable
name). They are never written to disk by this package.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

KEYCHAIN_SERVICE = "voicememos-mcp"

RECORDINGS_DIR_CANDIDATES = (
    "~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings",  # macOS 14+
    "~/Library/Application Support/com.apple.voicememos/Recordings",  # older macOS
)

DEFAULTS: dict[str, str] = {
    "VOICEMEMOS_RECORDINGS_DIR": "",
    "VOICEMEMOS_DATA_DIR": "~/Library/Application Support/voicememos-mcp",
    "VOICEMEMOS_OUTPUT_DIR": "~/Library/Application Support/voicememos-mcp/transcripts",
    "VOICEMEMOS_LANGUAGES": "en",
    "VOICEMEMOS_CONTEXT_TERMS": "",
    "VOICEMEMOS_CONTEXT_TEXT": "",
    "VOICEMEMOS_CONF_THRESHOLD": "0.70",
    "VOICEMEMOS_RUN_MIN": "4",
    "VOICEMEMOS_PAUSE_MS": "2500",
    "VOICEMEMOS_TIMESTAMP_EVERY_S": "60",
    "VOICEMEMOS_EXCLUDE_TITLE_REGEX": "",
    "VOICEMEMOS_EXCLUDE_FOLDERS": "",
    "VOICEMEMOS_AUTO_TRANSCRIBE": "1",
    "VOICEMEMOS_SINCE": "",
    "VOICEMEMOS_MIN_AGE_S": "60",
    "VOICEMEMOS_MAX_DURATION_MIN": "298",
    "VOICEMEMOS_MAX_ATTEMPTS": "3",
    "VOICEMEMOS_MARKERS": "idea=Idea,task=Task,remember=Remember,question=Question",
    "VOICEMEMOS_MARKER_WAKE": "",
    "VOICEMEMOS_MARKER_END": "",
    "VOICEMEMOS_WORKER": "launchd",
    "VOICEMEMOS_LAUNCHD_LABEL": "local.voicememos-mcp.sync",
    "SONIOX_BASE_URL": "https://api.soniox.com",
    "SONIOX_MODEL": "stt-async-v5",
    "VOICEMEMOS_DIARIZATION": "1",
    "VOICEMEMOS_VOICEPRINT_NAME": "",
    "VOICEMEMOS_VOICEPRINT_MODEL": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced",
    "VOICEMEMOS_SECOND_READER": "",
    "VOICEMEMOS_SECOND_READER_MODEL": "gemini-4-argon",
    "VOICEMEMOS_SECOND_READER_MAX_MIN": "60",
    "VOICEMEMOS_ROAM_ENABLED": "0",
    "VOICEMEMOS_ROAM_BACKFILL": "0",
    "ROAM_GRAPH": "",
    "VOICEMEMOS_ROAM_HEADING": "🎙️ Voice memo {time} ({duration}) #voicememo {stats}{title}",
}

PREFIXES = ("VOICEMEMOS_", "SONIOX_", "ROAM_", "GEMINI_")


def default_config_path() -> Path:
    return Path(os.environ.get("VOICEMEMOS_CONFIG", "~/.config/voicememos-mcp/config.env")).expanduser()


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY="value" file. Comments (#) and blank lines are ignored."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_markers(value: str) -> dict[str, str]:
    """``"idea=Idea,task=Task"`` -> ``{"idea": "Idea", "task": "Task"}`` (keys lower-cased)."""
    markers: dict[str, str] = {}
    for item in _split_list(value):
        spoken, _, label = item.partition("=")
        spoken = spoken.strip().lower()
        if spoken:
            markers[spoken] = (label.strip() or spoken.capitalize())
    return markers


@dataclass
class Config:
    raw: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    # ---- generic access
    def get(self, key: str) -> str:
        return self.raw.get(key, DEFAULTS.get(key, ""))

    def flag(self, key: str) -> bool:
        return self.get(key).strip().lower() in {"1", "true", "yes", "on"}

    # ---- paths
    @property
    def data_dir(self) -> Path:
        return Path(_expand(self.get("VOICEMEMOS_DATA_DIR")))

    @property
    def output_dir(self) -> Path:
        return Path(_expand(self.get("VOICEMEMOS_OUTPUT_DIR")))

    @property
    def recordings_dir(self) -> Path:
        explicit = self.get("VOICEMEMOS_RECORDINGS_DIR")
        if explicit:
            return Path(_expand(explicit))
        for candidate in RECORDINGS_DIR_CANDIDATES:
            path = Path(candidate).expanduser()
            try:
                if path.exists():
                    return path
            except PermissionError:
                return path
        return Path(RECORDINGS_DIR_CANDIDATES[0]).expanduser()

    # ---- transcription
    @property
    def languages(self) -> list[str]:
        return _split_list(self.get("VOICEMEMOS_LANGUAGES")) or ["en"]

    @property
    def context_terms(self) -> list[str]:
        return _split_list(self.get("VOICEMEMOS_CONTEXT_TERMS"))

    @property
    def context_text(self) -> str:
        return self.get("VOICEMEMOS_CONTEXT_TEXT")

    @property
    def threshold(self) -> float:
        return float(self.get("VOICEMEMOS_CONF_THRESHOLD"))

    @property
    def run_min(self) -> int:
        return int(self.get("VOICEMEMOS_RUN_MIN"))

    @property
    def pause_ms(self) -> int:
        return int(self.get("VOICEMEMOS_PAUSE_MS"))

    @property
    def timestamp_every_s(self) -> int:
        return max(0, int(self.get("VOICEMEMOS_TIMESTAMP_EVERY_S") or 0))

    @property
    def markers(self) -> dict[str, str]:
        return _parse_markers(self.get("VOICEMEMOS_MARKERS"))

    @property
    def marker_spec(self):
        from .notation import MarkerSpec

        return MarkerSpec(kinds=self.markers,
                          wake=[w.lower() for w in _split_list(self.get("VOICEMEMOS_MARKER_WAKE"))],
                          end=[w.lower() for w in _split_list(self.get("VOICEMEMOS_MARKER_END"))])

    # ---- selection
    @property
    def exclude_title_regex(self) -> str:
        return self.get("VOICEMEMOS_EXCLUDE_TITLE_REGEX")

    @property
    def exclude_folders(self) -> set[str]:
        return {name.casefold() for name in _split_list(self.get("VOICEMEMOS_EXCLUDE_FOLDERS"))}

    @property
    def since(self) -> str:
        return self.get("VOICEMEMOS_SINCE")

    @property
    def min_age_s(self) -> float:
        return float(self.get("VOICEMEMOS_MIN_AGE_S"))

    @property
    def max_duration_s(self) -> float:
        return float(self.get("VOICEMEMOS_MAX_DURATION_MIN")) * 60

    @property
    def max_attempts(self) -> int:
        return int(self.get("VOICEMEMOS_MAX_ATTEMPTS"))

    # ---- secrets
    def secret(self, name: str) -> str | None:
        return get_secret(name, self.raw)


def get_secret(name: str, raw: dict[str, str] | None = None) -> str | None:
    """Environment -> config dict -> macOS Keychain. Returns None when not found."""
    value = os.environ.get(name) or (raw or {}).get(name)
    if value:
        return value
    if platform.system() == "Darwin" and shutil.which("security"):
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> Config:
    path = path or default_config_path()
    raw = dict(DEFAULTS)
    raw.update(parse_env_file(path))
    environ = os.environ if env is None else env
    for key, value in environ.items():
        if key.startswith(PREFIXES):
            raw[key] = value
    return Config(raw=raw, source=path if path.exists() else None)
