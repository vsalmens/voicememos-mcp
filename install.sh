#!/bin/bash
# Install or update voicememos-mcp on macOS. Run from the repository:  ./install.sh
#
#   1. MCP server      -> uv tool (~/.local/bin/voicememos-mcp), used by Claude Desktop
#   2. worker          -> its own virtualenv, run by launchd; the only part that reads Voice Memos.
#                         A small compiled launcher (launcher/launcher.c) starts it, so Full
#                         Disk Access is granted to that launcher once and survives Python upgrades.
#   3. configuration   -> ~/.config/voicememos-mcp/config.env (created from the example if missing)
#   4. secrets         -> macOS Keychain, service "voicememos-mcp" (never written to a file):
#                         SONIOX_API_KEY, plus GEMINI_API_KEY / ROAM_API_TOKEN when those features are on
#   5. Claude Desktop  -> adds a "voicememos" entry to claude_desktop_config.json (backup kept)
#
# Options:  --no-claude-config   do not touch the Claude Desktop config
#           --no-launchd         do not install the background worker
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.config/voicememos-mcp"
CONFIG="${VOICEMEMOS_CONFIG:-$CONFIG_DIR/config.env}"
SERVICE="voicememos-mcp"
WORKER_PYTHON_VERSION="${VOICEMEMOS_WORKER_PYTHON:-3.13}"
DO_CLAUDE=1
DO_LAUNCHD=1
for arg in "$@"; do
  case "$arg" in
    --no-claude-config) DO_CLAUDE=0 ;;
    --no-launchd) DO_LAUNCHD=0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
[ "$(uname)" = "Darwin" ] || echo "warning: Voice Memos only exists on macOS"

echo "== 1/6 configuration"
mkdir -p "$CONFIG_DIR" && chmod 700 "$CONFIG_DIR"
if [ ! -f "$CONFIG" ]; then
  cp "$SRC_DIR/config.example.env" "$CONFIG" && chmod 600 "$CONFIG"
  echo "  created $CONFIG (edit it, then run ./install.sh again)"
else
  echo "  keeping $CONFIG"
fi
read_cfg() {  # read_cfg KEY DEFAULT  (value from the config file, ~ expanded)
  local v
  v="$(grep -E "^[[:space:]]*(export[[:space:]]+)?$1=" "$CONFIG" | tail -1 | sed -E 's/^[^=]*=//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/')"
  v="${v:-$2}"
  echo "${v/#\~/$HOME}"
}
DATA_DIR="$(read_cfg VOICEMEMOS_DATA_DIR "$HOME/Library/Application Support/voicememos-mcp")"
LABEL="$(read_cfg VOICEMEMOS_LAUNCHD_LABEL local.voicememos-mcp.sync)"
ROAM_ENABLED="$(read_cfg VOICEMEMOS_ROAM_ENABLED 0)"
mkdir -p "$DATA_DIR/queue" "$DATA_DIR/jobs" "$DATA_DIR/logs"

echo "== 2/6 MCP server (uv tool)"
uv tool install --force --reinstall "$SRC_DIR" >/dev/null
SERVER_BIN="$HOME/.local/bin/voicememos-mcp"
echo "  $SERVER_BIN ($("$SERVER_BIN" --version))"

echo "== 3/6 worker environment"
WORKER_VENV="$DATA_DIR/worker-venv"
uv venv --allow-existing --python "$WORKER_PYTHON_VERSION" "$WORKER_VENV" >/dev/null
# The worker also gets the optional voiceprint dependencies (local speaker model, no torch).
uv pip install --python "$WORKER_VENV/bin/python" --reinstall "$SRC_DIR[voiceprint]" >/dev/null
WORKER_PY="$WORKER_VENV/bin/python"
echo "  $WORKER_PY"

# Fixed-command launcher (launcher/, same as the fda-launcher template): rebuilt only when its
# source, the command or the signing identity changes, so the binary and its Full Disk Access
# grant survive updates. FDA_LAUNCHER_SIGN_IDENTITY (optional) is passed through.
LAUNCHER="$DATA_DIR/bin/voicememos-worker"
if "$SRC_DIR/launcher/build-launcher.sh" --name voicememos-mcp --out "$LAUNCHER" \
     -- "$WORKER_PY" -m voicememos_mcp sync >/dev/null; then
  FDA_TARGET="$LAUNCHER"
elif [ -x "$LAUNCHER" ]; then
  # Never fall back to the interpreter while a working launcher exists: its grant would be lost.
  echo "  launcher not rebuilt (see the message above); keeping the existing one."
  FDA_TARGET="$LAUNCHER"
else
  echo "  no launcher (needs a C compiler: xcode-select --install); falling back to the interpreter."
  echo "  Full Disk Access will then have to be granted again after Python upgrades."
  LAUNCHER=""
  FDA_TARGET="$("$WORKER_PY" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
fi

echo "== 4/6 secrets (Keychain service: $SERVICE)"
store_secret() {
  if security find-generic-password -s "$SERVICE" -a "$1" -w >/dev/null 2>&1; then
    echo "  $1: already stored"
  else
    read -r -p "  Store $1 in the Keychain now? [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
      # -w as the last argument makes `security` prompt for the value, so it never appears in `ps`.
      security add-generic-password -U -s "$SERVICE" -a "$1" -w && echo "  $1: stored"
    fi
  fi
}
store_secret SONIOX_API_KEY
[ "$ROAM_ENABLED" = "1" ] && store_secret ROAM_API_TOKEN
store_secret GEMINI_API_KEY   # optional: only needed for the second reader
true

if [ "$DO_LAUNCHD" = "1" ]; then
  echo "== 5/6 launchd worker ($LABEL)"
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  if [ -n "$LAUNCHER" ]; then
    "$WORKER_PY" -m voicememos_mcp launchd-plist --launcher "$LAUNCHER" > "$PLIST"
  else
    "$WORKER_PY" -m voicememos_mcp launchd-plist --python "$WORKER_PY" > "$PLIST"
  fi
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "  installed $PLIST"
else
  echo "== 5/6 launchd worker: skipped"
fi

if [ "$DO_CLAUDE" = "1" ]; then
  echo "== 6/6 Claude Desktop config"
  "$WORKER_PY" - "$SERVER_BIN" <<'PY'
import json, shutil, sys
from pathlib import Path
path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
if not path.exists():
    print(f"  {path} not found; add the server manually (see README)")
    sys.exit(0)
shutil.copy2(path, path.with_suffix(".json.bak"))
config = json.loads(path.read_text(encoding="utf-8"))
config.setdefault("mcpServers", {})["voicememos"] = {"command": sys.argv[1], "args": ["serve"]}
path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"  'voicememos' entry written (backup: {path.with_suffix('.json.bak').name}); restart Claude Desktop")
PY
else
  echo "== 6/6 Claude Desktop config: skipped"
fi

if [ "$DO_LAUNCHD" = "1" ]; then
  echo
  echo "Checking whether the worker can read Voice Memos ..."
  BEFORE="$(date +%s)"
  # No -k: a run that is already transcribing is left alone; its status answers the question too.
  launchctl kickstart "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
  # Wait for a worker run that started after the kickstart, or one that is still running (usually a
  # second or two, at most 30 s), instead of a fixed pause that could read the previous run's result.
  "$WORKER_PY" - "$DATA_DIR/worker.json" "$BEFORE" <<'PY' 2>/dev/null
import datetime as dt, json, sys, time
path, before = sys.argv[1], float(sys.argv[2])
for _ in range(60):
    try:
        d = json.load(open(path))
        started = dt.datetime.fromisoformat(d.get("started", "")).timestamp()
        if d.get("running") or (started >= before - 1 and (d.get("finished") or d.get("error"))):
            break
    except (OSError, ValueError):
        pass
    time.sleep(0.5)
PY
  if "$WORKER_PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('library_readable') else 1)" \
       "$DATA_DIR/worker.json" 2>/dev/null; then
    echo "  ok: the worker can read the library. Log: $DATA_DIR/logs/worker.log"
  else
    echo "  The worker cannot read Voice Memos yet. Grant Full Disk Access to:"
    echo "    $FDA_TARGET"
    echo "  System Settings > Privacy & Security > Full Disk Access > + > Cmd+Shift+G, paste the path."
    echo "  Then:  launchctl kickstart -k gui/$(id -u)/$LABEL"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null || true
  fi
fi
