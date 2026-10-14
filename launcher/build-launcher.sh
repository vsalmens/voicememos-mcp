#!/bin/bash
# Build a fixed-command launcher for one background job (and optionally a launchd job for it).
#
#   build-launcher.sh --name NAME [--out PATH] [launchd options] -- /absolute/command [args...]
#
# Options:
#   --name NAME        job name: lower case letters, digits and dashes
#   --out PATH         launcher path (default: ~/Library/Application Support/NAME/bin/NAME-launcher)
#   --launchd LABEL    also write ~/Library/LaunchAgents/LABEL.plist running the launcher, and load it
#   --watch PATH       launchd: run when PATH changes (repeatable)
#   --interval SEC     launchd: also run every SEC seconds
#   --log-dir DIR      launchd: stdout/stderr logs (default: ~/Library/Logs/NAME)
#
# Environment:
#   FDA_LAUNCHER_NO_LOAD=1             write the plist without loading it
#   FDA_LAUNCHER_SIGN_IDENTITY=NAME    sign with this code signing identity from the Keychain
#                                      instead of ad hoc, so grants survive rebuilds (see README).
#                                      Remembered next to the launcher; "-" switches back to ad hoc.
#
# The launcher is rebuilt only when launcher.c, the command or the signing identity changes,
# so an existing privacy grant (Full Disk Access) stays valid across updates. The last line
# printed to stdout is the launcher path; messages go to stderr.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/launcher.c"
NAME="" OUT="" LABEL="" INTERVAL="" LOG_DIR=""
WATCH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --launchd) LABEL="$2"; shift 2 ;;
    --watch) WATCH+=("$2"); shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "--name must be lower case letters, digits and dashes" >&2; exit 2; }
[ $# -ge 1 ] || { echo "missing command after --" >&2; exit 2; }
[[ "$1" == /* ]] || { echo "the command must be an absolute path (no PATH lookup): $1" >&2; exit 2; }
[ -x "$1" ] || echo "warning: $1 is not executable (yet)" >&2
[ -z "$INTERVAL" ] || [[ "$INTERVAL" =~ ^[0-9]+$ ]] || { echo "--interval must be a number" >&2; exit 2; }
[ -z "$LABEL" ] || [[ "$LABEL" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "invalid launchd label" >&2; exit 2; }
OUT="${OUT:-$HOME/Library/Application Support/$NAME/bin/$NAME-launcher}"

# C string literals for every argument; newlines are not allowed.
ARGV_C=""
for arg in "$@"; do
  case "$arg" in *$'\n'*) echo "arguments must not contain newlines" >&2; exit 2 ;; esac
  escaped="${arg//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  ARGV_C+="${ARGV_C:+,}\"$escaped\""
done

# Signing identity: from the environment, else the one used for the previous build.
ID_FILE="$OUT.sign-identity"
SIGN_ID="${FDA_LAUNCHER_SIGN_IDENTITY:-}"
if [ -z "$SIGN_ID" ] && [ -f "$ID_FILE" ]; then SIGN_ID="$(cat "$ID_FILE")"; fi
[ "$SIGN_ID" != "-" ] || SIGN_ID=""
case "$SIGN_ID" in *$'\n'*) echo "invalid signing identity" >&2; exit 2 ;; esac

STAMP="$OUT.stamp"
WANT="$( { cat "$SRC"; printf '%s\0' "$@"; [ -z "$SIGN_ID" ] || printf 'sign:%s\0' "$SIGN_ID"; } \
  | shasum -a 256 | cut -d' ' -f1)"
if [ -x "$OUT" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ]; then
  echo "launcher unchanged: existing permission grants stay valid" >&2
else
  command -v cc >/dev/null || { echo "no C compiler; install it with: xcode-select --install" >&2; exit 1; }
  mkdir -p "$(dirname "$OUT")"
  rm -f "$OUT.tmp"
  cc -O2 -Wall -o "$OUT.tmp" "-DLAUNCH_ARGV=$ARGV_C" "$SRC"
  if [ -n "$SIGN_ID" ]; then
    # A fixed identifier plus a certificate gives a designated requirement that does not
    # depend on the binary's hash, so the grant carries over to the rebuilt launcher.
    if ! command -v codesign >/dev/null \
       || ! codesign --force --sign "$SIGN_ID" --identifier "local.fda-launcher.$NAME" "$OUT.tmp"; then
      rm -f "$OUT.tmp"
      echo "signing with identity \"$SIGN_ID\" failed; the existing launcher was left unchanged" >&2
      echo "(create the certificate first, see README; or set FDA_LAUNCHER_SIGN_IDENTITY=- for ad hoc)" >&2
      exit 1
    fi
  fi
  mv "$OUT.tmp" "$OUT"
  echo "$WANT" > "$STAMP"
  if [ -n "$SIGN_ID" ]; then echo "$SIGN_ID" > "$ID_FILE"; else rm -f "$ID_FILE"; fi
  if [ -n "$SIGN_ID" ]; then
    echo "launcher built and signed with \"$SIGN_ID\": a grant made to a launcher signed with this" >&2
    echo "identity stays valid; otherwise grant Full Disk Access once and restart the job" >&2
  else
    echo "launcher built: grant it Full Disk Access (System Settings > Privacy & Security), then restart the job" >&2
  fi
fi

if [ -n "$LABEL" ]; then
  xml() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }
  LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/$NAME}"
  mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "  <key>Label</key><string>$(xml "$LABEL")</string>"
    echo "  <key>ProgramArguments</key><array><string>$(xml "$OUT")</string></array>"
    if [ ${#WATCH[@]} -gt 0 ]; then
      printf '  <key>WatchPaths</key><array>'
      for w in "${WATCH[@]}"; do printf '<string>%s</string>' "$(xml "$w")"; done
      echo '</array>'
    fi
    [ -z "$INTERVAL" ] || echo "  <key>StartInterval</key><integer>$INTERVAL</integer>"
    echo '  <key>RunAtLoad</key><true/>'
    echo '  <key>ProcessType</key><string>Background</string>'
    echo "  <key>StandardOutPath</key><string>$(xml "$LOG_DIR/$NAME.log")</string>"
    echo "  <key>StandardErrorPath</key><string>$(xml "$LOG_DIR/$NAME.err.log")</string>"
    echo '</dict></plist>'
  } > "$PLIST"
  if [ -z "${FDA_LAUNCHER_NO_LOAD:-}" ] && command -v launchctl >/dev/null; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "launchd job loaded: $LABEL" >&2
  fi
fi

echo "$OUT"
