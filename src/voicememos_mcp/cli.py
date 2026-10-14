"""Command line entry point: ``voicememos-mcp <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from . import __version__, notation
from .audio import find_tool, prepare_audio
from .config import KEYCHAIN_SERVICE, load_config
from .library import LibraryError, is_excluded, read_library


def cmd_serve(_args) -> int:
    from .server import main as serve

    serve()
    return 0


def cmd_sync(args) -> int:
    from .worker import Worker

    cfg = load_config()
    summary = Worker(cfg).run(dry_run=args.dry_run, limit=args.limit, only=args.memo, redo=args.redo)
    if args.dry_run:
        for item in summary.get("transcribed", []):
            print(f"== {item.get('recorded', '')[:16]} {item.get('title') or ''}")
            print(notation.stats_line(item["stats"]))
            print(item.get("text", ""))
            print()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("transcribed",)}
                     | {"transcribed": [i.get("short_id") for i in summary.get("transcribed", [])]},
                     ensure_ascii=False, indent=1, default=str))
    return 0 if summary.get("ok") else 1


def cmd_second_read(args) -> int:
    from .worker import Worker

    cfg = load_config()
    if args.queue:
        from .tools import Tools

        since = args.since or ("1970-01-01" if not args.memo else None)
        print(json.dumps(Tools(cfg).second_read(args.memo, since, args.until, args.model, args.redo, args.limit),
                         indent=1))
        return 0
    summary = Worker(cfg).run_second_read(only=args.memo, since=args.since, until=args.until, limit=args.limit,
                                          model=args.model, redo=args.redo)
    for item in summary.get("items", []):
        agreement = (item.get("agreement") or {}).get("agreement")
        extra = f" agreement {agreement:.2f}" if agreement is not None else f" {item.get('error') or ''}"
        print(f"{item.get('recorded', '')[:16]}  {item['short_id']}  {item['status']}{extra}  {item.get('title') or ''}")
    if not summary.get("ok"):
        print(summary.get("error"), file=sys.stderr)
        return 1
    return 0


def cmd_report(args) -> int:
    from .report import build_report, write_report
    from .store import Store

    cfg = load_config()
    store = Store(cfg.data_dir, cfg.output_dir)
    paths = write_report(store, build_report(store, args.since, args.until, args.samples))
    print(paths["markdown"])
    return 0


def cmd_check(_args) -> int:
    cfg = load_config()
    ok = True
    print(f"voicememos-mcp {__version__}")
    print(f"config file:    {cfg.source or '(none, using defaults and environment)'}")
    print(f"python:         {Path(sys.executable).resolve()}")
    print(f"recordings:     {cfg.recordings_dir}")
    try:
        memos = read_library(cfg.recordings_dir)
        excluded = sum(1 for m in memos if is_excluded(m, cfg.exclude_title_regex, cfg.exclude_folders))
        local = sum(1 for m in memos if m.local)
        print(f"  ok: {len(memos)} memos, {local} with local audio, {excluded} excluded by config")
    except LibraryError as exc:
        ok = False
        print(f"  NOT READABLE: {exc}")
    print(f"data dir:       {cfg.data_dir}")
    print(f"output dir:     {cfg.output_dir}")
    key = cfg.secret("SONIOX_API_KEY")
    print(f"Soniox key:     {'found' if key else 'MISSING (environment or Keychain service ' + KEYCHAIN_SERVICE + ')'}")
    ok &= bool(key)
    print(f"Soniox:         {cfg.get('SONIOX_BASE_URL')} model {cfg.get('SONIOX_MODEL')} languages {cfg.languages}")
    print(f"ffmpeg:         {find_tool('ffmpeg') or '(not found; afconvert is used for .qta)'}")
    reader = cfg.get("VOICEMEMOS_SECOND_READER")
    gemini_key = cfg.secret("GEMINI_API_KEY")
    print(f"second reader:  {reader or 'off'} ({cfg.get('VOICEMEMOS_SECOND_READER_MODEL')}), "
          f"Gemini key {'found' if gemini_key else 'missing'}")
    if reader:
        ok &= bool(gemini_key)
    if cfg.flag("VOICEMEMOS_ROAM_ENABLED"):
        token = cfg.secret("ROAM_API_TOKEN")
        roam_ok = bool(token and cfg.get("ROAM_GRAPH"))
        print(f"Roam:           {'configured' if roam_ok else 'MISSING ROAM_GRAPH or ROAM_API_TOKEN'}")
        ok &= roam_ok
    else:
        print("Roam:           disabled")
    return 0 if ok else 1


def cmd_transcribe_file(args) -> int:
    """Transcribe one audio file and print the result. Nothing is saved. Useful for calibration."""
    from .worker import soniox_transcriber

    cfg = load_config()
    path = Path(args.file).expanduser()
    with tempfile.TemporaryDirectory(prefix="voicememos-") as tmp:
        tokens = soniox_transcriber(cfg)(prepare_audio(path, Path(tmp)))
    words, had_confidence = notation.tokens_to_words(tokens)
    if args.json:
        print(json.dumps({"words": words, "source_tokens": tokens}, ensure_ascii=False, indent=1))
        return 0
    for threshold in args.threshold or [cfg.threshold]:
        print(f"== threshold {threshold}: {notation.stats_line(notation.stats(words, threshold))}")
        if not had_confidence:
            print("(no confidence values returned)")
        print("\n\n".join(notation.render_blocks(words, threshold, cfg.run_min, cfg.pause_ms)))
        print()
    return 0


def launchd_plist(cfg, program: list[str]) -> str:
    logs = cfg.data_dir / "logs"
    args = "".join(f"<string>{escape(p)}</string>" for p in program)
    watch = "".join(f"<string>{escape(str(p))}</string>" for p in (cfg.recordings_dir, cfg.data_dir / "queue"))
    path_env = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{escape(cfg.get('VOICEMEMOS_LAUNCHD_LABEL'))}</string>
  <key>ProgramArguments</key><array>{args}</array>
  <key>WatchPaths</key><array>{watch}</array>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>ProcessType</key><string>Background</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>{path_env}</string></dict>
  <key>StandardOutPath</key><string>{escape(str(logs / 'worker.log'))}</string>
  <key>StandardErrorPath</key><string>{escape(str(logs / 'worker.err.log'))}</string>
</dict></plist>
"""


def cmd_launchd_plist(args) -> int:
    cfg = load_config()
    if args.launcher:
        program = [args.launcher]
    else:
        program = [args.python or sys.executable, "-m", "voicememos_mcp", "sync"]
    print(launchd_plist(cfg, program), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicememos-mcp", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="run the MCP server on stdio (default)")

    p = sub.add_parser("sync", help="worker run: refresh library, process queue, transcribe new memos")
    p.add_argument("--dry-run", action="store_true", help="print results, save nothing")
    p.add_argument("--limit", type=int, help="transcribe at most N memos")
    p.add_argument("--memo", action="append", help="only this memo id (repeatable)")
    p.add_argument("--redo", action="store_true", help="with --memo: transcribe again")

    sub.add_parser("check", help="check paths, permissions and keys")

    p = sub.add_parser("second-read", help="add the second reader (Gemini) to existing transcripts")
    p.add_argument("--memo", action="append", help="only this memo id (repeatable)")
    p.add_argument("--since", help="memos recorded on or after YYYY-MM-DD")
    p.add_argument("--until", help="memos recorded on or before YYYY-MM-DD")
    p.add_argument("--limit", type=int, help="at most N memos")
    p.add_argument("--model", help="model id (default VOICEMEMOS_SECOND_READER_MODEL)")
    p.add_argument("--redo", action="store_true", help="read again even if already read")
    p.add_argument("--queue", action="store_true",
                   help="hand the request to the launchd worker (use when this shell lacks Full Disk Access)")

    p = sub.add_parser("report", help="compare the readers, per language; writes OUTPUT_DIR/_reports/")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--samples", type=int, default=15, help="disagreements to sample per language")

    p = sub.add_parser("transcribe-file", help="transcribe one audio file and print it (nothing saved)")
    p.add_argument("file")
    p.add_argument("--threshold", type=float, action="append", help="render with this threshold (repeatable)")
    p.add_argument("--json", action="store_true", help="print word and raw token data")

    p = sub.add_parser("launchd-plist", help="print a launchd job definition for the worker")
    p.add_argument("--python", help="interpreter for the job (default: this one)")
    p.add_argument("--launcher", help="run this fixed-command launcher instead of the interpreter")

    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("VOICEMEMOS_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if platform.system() != "Darwin" and args.command in (None, "serve"):
        logging.getLogger(__name__).warning("Voice Memos is a macOS app; only transcripts can be served here")
    handlers = {None: cmd_serve, "serve": cmd_serve, "sync": cmd_sync, "check": cmd_check,
                "transcribe-file": cmd_transcribe_file, "launchd-plist": cmd_launchd_plist,
                "second-read": cmd_second_read, "report": cmd_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
