"""MCP server (stdio). Works with the ``mcp`` Python SDK 1.x (FastMCP) and 2.x (MCPServer)."""

from __future__ import annotations

import logging
from typing import Optional

from . import __version__
from .config import load_config
from .notation import READER_NOTE
from .tools import ToolError, Tools

log = logging.getLogger(__name__)

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = (
    "Access to the user's Apple Voice Memos and their transcripts. Use voicememos_list or "
    "voicememos_search to find memos, voicememos_read to read a transcript, and voicememos_transcribe "
    "for memos that are not transcribed yet. " + READER_NOTE
)


def build_server(tools: Tools | None = None):
    tools = tools or Tools(load_config())
    server = _Server("voicememos", instructions=INSTRUCTIONS)

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError as exc:
            return {"error": str(exc)}
        except PermissionError as exc:
            log.exception("permission denied in %s", getattr(fn, "__name__", fn))
            return {"error": f"permission denied: {exc}. If the path is in Documents, Desktop, Downloads or "
                             "iCloud Drive, allow the app that runs this server in System Settings > Privacy & "
                             "Security > Files and Folders, or move the folder."}
        except Exception as exc:  # noqa: BLE001 - report the cause instead of a bare "Error executing tool"
            log.exception("unexpected error in %s", getattr(fn, "__name__", fn))
            return {"error": f"internal error ({type(exc).__name__}): {exc}"}

    @server.tool()
    def voicememos_status() -> dict:
        """Library size, transcription status counts, worker health (alerts: what needs a person),
        usage per month (audio minutes and tokens sent to the engines) and configuration."""
        return call(tools.status)

    @server.tool()
    def voicememos_list(limit: int = 20, offset: int = 0, since: Optional[str] = None,
                        until: Optional[str] = None, title: Optional[str] = None,
                        status: Optional[str] = None) -> dict:
        """List memos, newest first.

        Args:
            limit: page size (max 200).
            offset: start index for paging.
            since: only memos recorded on or after this date (YYYY-MM-DD).
            until: only memos recorded on or before this date (YYYY-MM-DD).
            title: case-insensitive substring of the title.
            status: transcribed, pending, segmenting (longer than the engine accepts: being
                transcribed in parts), not_transcribed (outside VOICEMEMOS_SINCE, or the title contains the
                skip keyword, default 'notranscript'), not_downloaded, failed (several tries failed;
                tried again once a day), too_long (long and cannot be split: splitting off or ffmpeg
                missing) or audio_removed (transcribed, the recording itself is gone; still readable).
        """
        return call(tools.list_memos, limit, offset, since, until, title, status)

    @server.tool()
    def voicememos_read(memo: str, format: str = "markdown", threshold: Optional[float] = None,
                        offset: int = 0, max_chars: int = 20000) -> dict:
        """Read one transcript with uncertainty marks.

        Args:
            memo: memo id from voicememos_list (8 characters is enough).
            format: 'markdown' (with header), 'text' (transcript only), 'json' (word-level data),
                'ca' (a light conversation-analysis view: pauses, gaps between turns and, when the
                recording has been measured, pitch jumps, loudness, tempo and stretched sounds),
                'vtt' (WebVTT subtitles) or 'tei' (TEI XML, spoken text).
            threshold: confidence threshold for marking words (default from config); re-renders
                without re-transcribing.
            offset: character offset for long transcripts (see next_offset).
            max_chars: maximum characters returned (1000-100000).
        """
        return call(tools.read, memo, format, threshold, offset, max_chars)

    @server.tool()
    def voicememos_search(query: str, limit: int = 20, since: Optional[str] = None,
                          until: Optional[str] = None, regex: bool = False) -> dict:
        """Search transcripts and titles (case-insensitive). Returns snippets per memo.

        Args:
            query: text to find, or a regular expression when regex=true.
            limit: maximum number of memos returned.
            since: only memos recorded on or after this date (YYYY-MM-DD).
            until: only memos recorded on or before this date (YYYY-MM-DD).
            regex: treat query as a regular expression.
        """
        return call(tools.search, query, limit, since, until, regex)

    @server.tool()
    def voicememos_markers(since: Optional[str] = None, until: Optional[str] = None,
                           label: Optional[str] = None, limit: int = 50) -> dict:
        """Sentences the user tagged by starting them with a spoken marker word (e.g. "Idea: ...").

        Args:
            since: only memos recorded on or after this date (YYYY-MM-DD).
            until: only memos recorded on or before this date (YYYY-MM-DD).
            label: only this marker label.
            limit: maximum number of markers returned.
        """
        return call(tools.markers, since, until, label, limit)

    @server.tool()
    def voicememos_transcribe(memos: Optional[list[str]] = None, since: Optional[str] = None,
                              until: Optional[str] = None, redo: bool = False,
                              push_roam: bool = False) -> dict:
        """Queue memos for transcription in the background. Returns a job_id immediately.

        Args:
            memos: memo ids to transcribe.
            since: or: all memos recorded on or after this date (YYYY-MM-DD).
            until: or: all memos recorded on or before this date (YYYY-MM-DD).
            redo: transcribe again even if a transcript exists.
            push_roam: also append the transcripts to Roam daily notes (if Roam is configured).
        """
        return call(tools.transcribe, memos, since, until, redo, push_roam)

    @server.tool()
    def voicememos_second_read(memos: Optional[list[str]] = None, since: Optional[str] = None,
                               until: Optional[str] = None, model: Optional[str] = None,
                               redo: bool = False, limit: Optional[int] = None,
                               only_uncertain: bool = False, realign: bool = False) -> dict:
        """Queue a second, independent reading (Gemini) of already transcribed memos.

        Disagreements with the primary engine appear as {a|b} in the transcript. Returns a job_id.

        Args:
            memos: memo ids.
            since: or: memos recorded on or after this date (YYYY-MM-DD).
            until: or: memos recorded on or before this date (YYYY-MM-DD).
            model: Gemini model id (default from config, e.g. gemini-4-argon).
            redo: read again even if a second reading exists.
            limit: at most this many memos (oldest first).
            only_uncertain: only memos with low certainty; long recordings (read in parts) only in
                their parts with low certainty.
            realign: no new reading; align the stored readings again with the current rules.
        """
        return call(tools.second_read, memos, since, until, model, redo, limit, only_uncertain, realign)

    @server.tool()
    def voicememos_compare_readers(since: Optional[str] = None, until: Optional[str] = None,
                                   samples: int = 15) -> dict:
        """Compare primary engine and second reader per language (agreement, calibration, samples).

        Writes a Markdown report with a listening checklist and returns the summary.

        Args:
            since: only memos recorded on or after this date (YYYY-MM-DD).
            until: only memos recorded on or before this date (YYYY-MM-DD).
            samples: disagreements to sample per language (max 50).
        """
        return call(tools.compare_readers, since, until, samples)

    @server.tool()
    def voicememos_voiceprint(action: str = "status", memos: Optional[list[str]] = None,
                              since: Optional[str] = None, until: Optional[str] = None,
                              name: Optional[str] = None, speaker: Optional[str] = None) -> dict:
        """Recognise the owner's voice with a local voiceprint.

        Args:
            action: 'status' (enrollment state, thresholds and suggested enrollment memos),
                'enroll' (build the voiceprint from memos where only the owner speaks),
                'identify' (apply it to already transcribed memos),
                'add_other' (store a known other voice, e.g. a relative who sounds similar, from
                one speaker of one memo; it is never shown by name, it only prevents mistaking
                that person for the owner), 'add_voice' (the same, but named: speakers who match
                it are labelled with the name in transcripts made or identified afterwards; add
                the same person from several recordings for a steadier match, and only store the
                voice of somebody who knows) or 'remove_other' / 'remove_voice' (by name).
            memos: memo ids for 'enroll' (2+, ideally 4-6 from different years), 'identify', or
                one memo for 'add_other' / 'add_voice'.
            since: for 'identify': memos recorded on or after this date (YYYY-MM-DD).
            until: for 'identify': memos recorded on or before this date (YYYY-MM-DD).
            name: the owner's name for 'enroll' (default from config); the other person's name
                for 'add_other' / 'add_voice' / 'remove_other' / 'remove_voice'.
            speaker: for 'add_other' / 'add_voice' in a multi-speaker memo: the speaker label as
                the transcript shows it (S1, S2, ...).
        """
        return call(tools.voiceprint, action, memos, since, until, name, speaker)

    @server.tool()
    def voicememos_annotate(what: list[str], memos: Optional[list[str]] = None, since: Optional[str] = None,
                            until: Optional[str] = None, redo: bool = False, limit: Optional[int] = None,
                            model: Optional[str] = None) -> dict:
        """Queue optional passes over already transcribed memos. Returns a job_id.

        Args:
            what: one or more of 'speakers' (divide and join the speech engine's speakers by
                voice on this machine, then apply the voiceprint and names again; the engine's own
                numbers stay in the data), 'prosody' (measure loudness, pitch and tempo per word
                against each speaker's own level; read them with voicememos_read format 'ca') and
                'adjudicate' (a third listening by another model of the spots where the two readers
                differ; needs a second reading; marks the reading it sided with as {a*|b}).
            memos: memo ids.
            since: or: memos recorded on or after this date (YYYY-MM-DD).
            until: or: memos recorded on or before this date (YYYY-MM-DD).
            redo: do a pass again even if the memo has been through it.
            limit: at most this many memos (oldest first).
            model: for 'adjudicate': the model id (default from config).
        """
        return call(tools.annotate, what, memos, since, until, redo, limit, model)

    @server.tool()
    def voicememos_transcribe_status(job_id: Optional[str] = None) -> dict:
        """Progress of a transcription job, or the 10 most recent jobs when job_id is omitted."""
        return call(tools.transcribe_status, job_id)

    return server


def main() -> None:
    build_server().run()


__all__ = ["build_server", "main", "__version__"]
