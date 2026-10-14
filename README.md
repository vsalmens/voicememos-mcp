# voicememos-mcp

An MCP server for Apple Voice Memos on macOS. It transcribes memos with
[Soniox](https://soniox.com) and keeps the speech engine's word-level confidence, so
uncertain words stay visibly marked instead of being silently guessed. Claude (or any
MCP client) can then list, read and search the transcripts.

Design goals:

* **A visible guess beats a gap, and a gap beats a confident wrong word.** Every word is
  kept; low-confidence words are marked.
* **Least privilege.** Only a small background worker reads the Voice Memos container. The
  MCP server reads transcript files and holds no secrets.
* **Lossless storage.** The engine's raw response is stored verbatim next to a canonical
  word-level JSON. Markdown is rendered from it, so the threshold can change later
  without paying for a new transcription.

## How it works

```
iPhone ──iCloud──▶ Voice Memos on the Mac
                        │  (Full Disk Access, read-only)
                        ▼
              worker (launchd)  ── audio ──▶ Soniox  (file and transcript deleted after use)
                        │
                        ├─▶ OUTPUT_DIR/2026/2026-10-13_2135_title_1a2b3c4d.{md,json}
                        ├─▶ DATA_DIR/library.json, state.json, jobs/
                        └─▶ Roam daily note (optional, append-only token)

Claude ──MCP──▶ voicememos-mcp serve  ── reads ──▶ library.json + transcripts
                        └── writes ──▶ DATA_DIR/queue/  (picked up by the worker)
```

The worker runs when the recordings folder or the queue changes, and every 30 minutes.

## Uncertainty notation

| Markdown | Meaning |
|---|---|
| `word (?)` | best guess, confidence below the threshold |
| `[?: w1 w2 w3 w4]` | several consecutive uncertain words (guesses kept) |
| `{a\|b}` | two engines heard different things: primary engine first, second reader after |
| `{a b\|∅}`, `{∅\|b}` | only one of the engines heard something here |
| `[?]` | something was said, nothing usable was recognised |

This follows conversation-analysis (Jefferson) practice, where `(word)` marks an
uncertain hearing, `(a)/(b)` alternative hearings and `( )` unhearable talk.

Each transcript's JSON keeps, per word, `conf` (0..1), optional ranked `alts`, and a
`basis` explaining where the uncertainty comes from:

* `asr_confidence`: measured by the speech engine,
* `reader_disagreement`: independent engines disagreed,
* `self_report`: a model said it was unsure (not calibrated).

The semantics match PAGE XML (`TextEquiv@conf`, alternatives ranked by `index`) and can be
exported to TEI (`<unclear cert>`, `<choice>`) or ALTO (`WC`, `ALTERNATIVE`). No existing
standard keeps everything a speech engine returns (sub-word tokens, per-token language,
timings), which is why the raw response is stored as well.

## Speakers

With `VOICEMEMOS_DIARIZATION=1` (default), Soniox separates speakers. Single-speaker memos
look the same as before; conversations get one paragraph per turn:

```
**S1:** How did it go?
**S2:** Fine (?), thanks.
```

Speakers are numbered in order of appearance; the labels only hold within one memo and nobody
is named. The JSON keeps per-speaker word counts and shares (`speakers`) and the speaker of
every word.

## Second reader

Gemini models return text without word confidence, so they cannot drive the `(?)` marks.
They are useful as an independent second reader: where two engines disagree, at least one is
wrong. `voicememos-mcp second-read` (or the `voicememos_second_read` tool) sends the audio
of already transcribed memos to a Gemini model (default `gemini-4-argon`), aligns its text
with the primary words and marks the differences as `{a|b}`. The primary engine is not
called again, and the raw second text is kept in the JSON (`readers`).

`voicememos-mcp report` (or `voicememos_compare_readers`) compares the readers **per
language** (Soniox's language tags decide whether a memo is Finnish, English or mixed):

* word agreement and difference rate,
* disagreements per 100 words,
* calibration: how many disagreements fall on words the primary engine already marked
  uncertain, and how many uncertain words the second reader confirmed,
* a random sample of disagreements with time offsets, as a listening checklist. Agreement
  says nothing about who is right; listening does.

The alignment is a word-level sequence diff that ignores case and punctuation. Different
word splitting (compounds, numbers written as digits or words) shows up as disagreement.

Transcripts start with a short reading guide that asks language models not to present
marked words as certain, and to propose corrections visibly (`word→suggestion (?)`).

## Requirements

* macOS 14 or later with Voice Memos iCloud sync enabled on the iPhone and the Mac
  (open Voice Memos on the Mac once).
* [uv](https://docs.astral.sh/uv/)
* A Soniox API key.
* Optional: `ffmpeg` for lossless `.qta` handling (otherwise `afconvert` is used).

## Install

```bash
git clone https://github.com/<you>/voicememos-mcp.git
cd voicememos-mcp
./install.sh          # creates ~/.config/voicememos-mcp/config.env on the first run
$EDITOR ~/.config/voicememos-mcp/config.env
./install.sh          # again, to apply the configuration
```

`install.sh` installs the MCP server with `uv tool`, creates a separate virtualenv for the
worker, stores the Soniox key in the Keychain, installs the launchd job and adds this to
`claude_desktop_config.json`:

```json
"voicememos": { "command": "/Users/<you>/.local/bin/voicememos-mcp", "args": ["serve"] }
```

### Full Disk Access

The Voice Memos container is protected by macOS privacy controls. Only the worker needs
access, and it gets it through a tiny compiled launcher
([`launcher/launcher.c`](launcher/launcher.c), built by `launcher/build-launcher.sh`) that `install.sh` builds
into `~/Library/Application Support/voicememos-mcp/bin/voicememos-worker`. Grant Full Disk
Access to that file once (`install.sh` prints the path and opens the settings pane), then run
`launchctl kickstart -k gui/$(id -u)/local.voicememos-mcp.sync`.

Why a launcher:

* macOS ties the permission to the exact binary. Granting it to a Python interpreter breaks
  with every Python update, and it would cover every script run with that interpreter.
* The launcher starts the worker as a child process, and macOS attributes the child's file
  access to it, as it does for programs started from Terminal. It runs one compiled-in
  command and ignores arguments, so it cannot be used to run anything else.
* `install.sh` rebuilds it only when its source or the worker path changes, so the grant
  survives updates. Without a C compiler (`xcode-select --install`) the installer falls back
  to granting the interpreter.

The MCP server does not need Full Disk Access. Do not grant it to Claude Desktop: every
local MCP server it starts would inherit it.

Check the setup with `voicememos-mcp check` (from Terminal; Terminal may have different
permissions than the launchd job) and the worker log in
`~/Library/Application Support/voicememos-mcp/logs/`.

## Configuration

All settings are in `~/.config/voicememos-mcp/config.env`; see
[`config.example.env`](config.example.env) for every option. The most important:

| Setting | Default | |
|---|---|---|
| `VOICEMEMOS_LANGUAGES` | `en` | language hints, e.g. `fi,en` |
| `VOICEMEMOS_CONTEXT_TERMS` | | names and rare terms the engine should know |
| `VOICEMEMOS_CONF_THRESHOLD` | `0.70` | words below this are marked |
| `VOICEMEMOS_OUTPUT_DIR` | `~/Documents/Voice Memo Transcripts` | |
| `VOICEMEMOS_SINCE` | | only auto-transcribe memos from this date on |
| `VOICEMEMOS_EXCLUDE_FOLDERS` | | Voice Memos folders hidden from everything |
| `VOICEMEMOS_EXCLUDE_TITLE_REGEX` | | titles hidden from everything |
| `VOICEMEMOS_MARKERS` | `idea=Idea,task=Task,...` | spoken words that tag a sentence |
| `VOICEMEMOS_DIARIZATION` | `1` | separate speakers in conversations |
| `VOICEMEMOS_SECOND_READER` | | `gemini` = second reading for every new memo |
| `VOICEMEMOS_SECOND_READER_MODEL` | `gemini-4-argon` | |
| `VOICEMEMOS_ROAM_ENABLED` | `0` | append new transcripts to Roam daily notes |

Excluded memos are never transcribed, and the MCP tools cannot see them, not even their
titles.

## MCP tools

| Tool | |
|---|---|
| `voicememos_status` | counts by status, worker health, configuration, warnings |
| `voicememos_list` | memos newest first; filter by date, title, status; paging |
| `voicememos_read` | one transcript as Markdown, plain text or word-level JSON; optional `threshold` re-renders without re-transcribing; paging for long memos |
| `voicememos_search` | case-insensitive search (or regex) over transcripts and titles, with snippets |
| `voicememos_markers` | sentences tagged with a spoken marker ("Idea: ...") across memos |
| `voicememos_transcribe` | queue memos (by id or date range) for background transcription; returns a job id |
| `voicememos_second_read` | queue an independent Gemini reading of transcribed memos |
| `voicememos_compare_readers` | per-language comparison of the readers, with a listening checklist |
| `voicememos_transcribe_status` | job progress, or recent jobs |

Memo status values: `transcribed`, `pending` (will be transcribed automatically),
`not_transcribed` (outside `VOICEMEMOS_SINCE`; use `voicememos_transcribe`),
`not_downloaded` (audio still in iCloud; play it once in Voice Memos on the Mac),
`failed` (after `VOICEMEMOS_MAX_ATTEMPTS` tries), `too_long`, `deleted`.

## Command line

```bash
voicememos-mcp check                         # paths, permissions, keys
voicememos-mcp sync --dry-run --limit 1      # one memo, print only, save nothing
voicememos-mcp sync --memo 1a2b3c4d --redo   # transcribe one memo again
voicememos-mcp transcribe-file memo.m4a --threshold 0.6 --threshold 0.8   # calibration
voicememos-mcp second-read --since 2026-01-01 --limit 20   # add the second reader
voicememos-mcp report                        # compare readers per language
voicememos-mcp launchd-plist                 # print the launchd job definition
```

## Calibrating the threshold

Pick 5-10 different recordings (quiet room, car, mixed languages) and run
`transcribe-file` with thresholds 0.6, 0.7 and 0.8. Choose the value at which wrong words
stop slipping through unmarked. Extra marks on correct words are cheap; an unmarked wrong
word is expensive. `voicememos_read(..., threshold=...)` lets you compare on stored
transcripts without new API calls.

## Privacy and security

* With the second reader, audio also goes to the Gemini API. Large files uploaded through
  the Files API are deleted right after the request.
* Audio goes to Soniox for transcription. The uploaded file and the transcription are
  deleted from Soniox right after the transcript is fetched, also when something fails.
  Soniox offers an EU endpoint (`SONIOX_BASE_URL=https://api.eu.soniox.com`), which needs an
  EU project.
* Nothing in the Voice Memos container is modified. The database is opened read-only.
* Secrets live in the Keychain or the environment. The package never writes them to a file
  (you can put them in `config.env` yourself, but the Keychain is the safer place).
* With the default `VOICEMEMOS_WORKER=launchd`, the MCP server has no secrets and no access
  to Voice Memos; it can only queue requests. In `inprocess` mode it does both itself.
* Roam uses an append-only token, which cannot read or delete anything.

## Limitations

* Tested with the Voice Memos database of macOS 26. Column names are detected at runtime,
  but older or newer versions may differ.
* Memos whose audio has not been downloaded from iCloud cannot be transcribed until they
  are played once on the Mac.
* Soniox limits a single file to about 300 minutes.
* The Roam Append API request format follows public examples; test with one memo first.

## Roadmap

* Adjudication: a third pass that listens to the disputed segment and picks a reading.
* Identifying the memo owner by voice (speaker embeddings compared with reference recordings).
* Export to TEI (spoken text) and WebVTT.

## Development

```bash
uv venv && uv pip install -e ".[test]"
.venv/bin/pytest
```

## License

MIT
