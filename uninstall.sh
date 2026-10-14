#!/bin/bash
# Remove the launchd worker, the uv tool and the Claude Desktop entry.
# Keeps: configuration, transcripts, state (delete those yourself if you want them gone).
set -uo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
CONFIG="${VOICEMEMOS_CONFIG:-$HOME/.config/voicememos-mcp/config.env}"
LABEL="$(grep -E '^VOICEMEMOS_LAUNCHD_LABEL=' "$CONFIG" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"')"
LABEL="${LABEL:-local.voicememos-mcp.sync}"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "stopped $LABEL"
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
uv tool uninstall voicememos-mcp 2>/dev/null && echo "removed uv tool"
DATA_DIR="$(grep -E '^VOICEMEMOS_DATA_DIR=' "$CONFIG" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"')"
DATA_DIR="${DATA_DIR:-$HOME/Library/Application Support/voicememos-mcp}"
DATA_DIR="${DATA_DIR/#\~/$HOME}"
rm -rf "$DATA_DIR/worker-venv" "$DATA_DIR/bin"

python3 - <<'PY'
import json
from pathlib import Path
path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
if path.exists():
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("mcpServers", {}).pop("voicememos", None) is not None:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("removed 'voicememos' from Claude Desktop config")
PY

echo "Kept: $CONFIG, transcripts and state."
echo "Remove secrets: security delete-generic-password -s voicememos-mcp -a SONIOX_API_KEY"
echo "                security delete-generic-password -s voicememos-mcp -a ROAM_API_TOKEN"
echo "Remove voicememos-worker from System Settings > Privacy & Security > Full Disk Access."
