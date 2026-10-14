"""Append a transcript to the Roam Research daily note of the recording date.

Uses Roam's Append API with an append-only token, which can add blocks but
cannot read or delete anything. Daily notes are addressed as ``MM-DD-YYYY``.
"""

from __future__ import annotations

import datetime as dt

APPEND_URL = "https://append-api.roamresearch.com/api/graph/{graph}/append-blocks"


class RoamError(RuntimeError):
    pass


def heading(template: str, memo: dict, stats_text: str) -> str:
    recorded = dt.datetime.fromisoformat(memo["recorded"]) if memo.get("recorded") else None
    duration = memo.get("duration_s") or 0
    title = memo.get("title")
    return template.format(
        time=f"{recorded:%H:%M}" if recorded else "",
        date=f"{recorded:%Y-%m-%d}" if recorded else "",
        duration=f"{int(duration // 60)}:{int(duration % 60):02d}",
        stats=f"({stats_text})" if stats_text else "",
        title=f" {title}" if title else "",
    ).strip()


def append_daily_note(graph: str, token: str, date: dt.date, parent: str, children: list[str],
                      session=None) -> None:
    if session is None:
        import requests

        session = requests
    body = {
        "location": {"page": {"title": {"daily-note-page": date.strftime("%m-%d-%Y")}}},
        "append-data": [{"string": parent, "children": [{"string": c} for c in children]}],
    }
    response = session.post(APPEND_URL.format(graph=graph), json=body,
                            headers={"X-Authorization": f"Bearer {token}"}, timeout=30)
    if not response.ok:
        raise RoamError(f"Roam {response.status_code}: {response.text[:300]}")
