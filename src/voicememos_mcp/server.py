"""MCP server (stdio). Works with the ``mcp`` Python SDK 1.x (FastMCP) and 2.x (MCPServer)."""

from __future__ import annotations

from typing import Optional

from . import __version__
from .config import load_config
from .notation import READER_NOTE
from .tools import ToolError, Tools

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

    @server.tool()
    def voicememos_status() -> dict:
        """Library size, transcription status counts, worker health and configuration."""
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
            status: transcribed, pending, not_transcribed, not_downloaded, failed or too_long.
        """
        return call(tools.list_memos, limit, offset, since, until, title, status)

    @server.tool()
    def voicememos_read(memo: str, format: str = "markdown", threshold: Optional[float] = None,
                        offset: int = 0, max_chars: int = 20000) -> dict:
        """Read one transcript with uncertainty marks.

        Args:
            memo: memo id from voicememos_list (8 characters is enough).
            format: 'markdown' (with header), 'text' (transcript only) or 'json' (word-level data).
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
    def voicememos_transcribe_status(job_id: Optional[str] = None) -> dict:
        """Progress of a transcription job, or the 10 most recent jobs when job_id is omitted."""
        return call(tools.transcribe_status, job_id)

    return server


def main() -> None:
    build_server().run()


__all__ = ["build_server", "main", "__version__"]
