"""A light conversation-analysis rendering, made from the word data on demand.

The Markdown transcript stays the readable version. This view adds how the talk went, from the
word times and the measurements of ``prosody.py``, in a small subset of Jefferson's notation:

=================  ==============================================================
``(0.8)``          a pause in seconds; ``(.)`` a short one
``=``              the next turn starts with no gap (end of one line, start of the next)
``↑word`` ``↓``    the pitch jumps up or down against the words just before
``WORD``           louder than the speaker's usual level, ``°word°`` quieter
``>words<``        faster than the speaker's usual tempo, ``<words>`` slower
``wo:rd``          a stretched sound (``::`` a long one)
=================  ==============================================================

One line is one turn: a change of speaker or a long pause starts a new one. The uncertainty
marks are the transcript's own (``word (?)``, ``{a|b}``, ``[?]``), so ``(word)`` is never used
for an uncertain hearing here. Loudness and tempo are marked for runs of words only, because a
single loud word is usually a stressed syllable and not a raised voice.

Soniox gives no overlapping talk and no audio events, so ``[`` and ``((laughs))`` cannot come
from it; and it tidies fillers and colloquial forms, so this is not a verbatim transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import notation

LEGEND = (
    "Conversation-analysis view: `(0.8)` pause in seconds, `(.)` a short pause, `=` no gap between "
    "turns, `↑word`/`↓word` pitch jump, `WORD` louder and `°word°` quieter than the speaker's usual, "
    "`>words<` faster and `<words>` slower, `wo:rd` stretched. Measured from the audio against each "
    "speaker's own level; not a reading of feeling. Uncertainty marks as in the transcript."
)


@dataclass
class Limits:
    """Where a measurement becomes a mark. Chosen on 26.11.2026 from four recordings (two
    meetings, an ambient recording, a dictation) so that each mark falls on about one to three
    words in a hundred: with +6 / -9 dB and no runs, loudness alone marked one word in five."""
    micro_pause_ms: int = 250    # (.) from this gap on
    pause_ms: int = 500          # (0.5) from this gap on
    turn_pause_ms: int = 2500    # a pause this long starts a new line
    latch_ms: int = 80           # a speaker change with at most this gap is latched
    loud_db: float = 10.0
    quiet_db: float = -15.0
    level_run: int = 2           # words in a row before loud/quiet is marked
    pitch_st: float = 7.0        # jump against the median of the words before
    pitch_context: int = 4       # that many words before, same speaker
    fast: float = 1.6
    slow: float = 0.6
    tempo_run: int = 3
    stretch_rate: float = 0.35   # a word at most this fast ...
    stretch_ms: int = 600        # ... and at least this long is stretched
    long_stretch_ms: int = 1000


_LAST_VOWELS = re.compile(r"([aeiouyäöåAEIOUYÄÖÅ]+)(?=[^aeiouyäöåAEIOUYÄÖÅ]*$)")


def _clock(ms: int, long_form: bool) -> str:
    sec = int(ms // 1000)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"[{h}:{m:02d}:{s:02d}]" if long_form else f"[{m:02d}:{s:02d}]"


def _pause(gap_ms: int | None, limits: Limits) -> str:
    if gap_ms is None or gap_ms < limits.micro_pause_ms:
        return ""
    return "(.)" if gap_ms < limits.pause_ms else f"({gap_ms / 1000:.1f})"


def _runs(flags: list[bool], same: list, minimum: int) -> list[bool]:
    """Keep only the True values that stand in a run of at least ``minimum`` with the same key."""
    out = [False] * len(flags)
    i = 0
    while i < len(flags):
        j = i
        while j < len(flags) and flags[j] and same[j] == same[i]:
            j += 1
        if flags[i] and j - i >= minimum:
            out[i:j] = [True] * (j - i)
        i = max(j, i + 1)
    return out


def _stretch(text: str, long: bool) -> str:
    body, tail = notation._split_trailing(text)
    mark = "::" if long else ":"
    stretched, n = _LAST_VOWELS.subn(lambda m: m.group(1) + mark, body, count=1)
    return (stretched if n else body + mark) + tail


def marks(words: list[dict], limits: Limits | None = None) -> list[dict]:
    """What would be marked on each word: ``{"loud", "quiet", "fast", "slow", "up", "down",
    "stretch"}`` (only the true ones). Also used to count how often each mark falls."""
    limits = limits or Limits()
    pros = [w.get("pros") or {} for w in words]
    spk = [str(w.get("speaker") or "") for w in words]
    loud = _runs([p.get("db", 0) >= limits.loud_db for p in pros], spk, limits.level_run)
    quiet = _runs([p.get("db", 0) <= limits.quiet_db for p in pros], spk, limits.level_run)
    fast = _runs([p.get("rate", 1) >= limits.fast for p in pros], spk, limits.tempo_run)
    slow = _runs([p.get("rate", 1) <= limits.slow for p in pros], spk, limits.tempo_run)
    out = []
    recent: dict[str, list[float]] = {}
    for k, word in enumerate(words):
        mark = {}
        if loud[k]:
            mark["loud"] = True
        if quiet[k]:
            mark["quiet"] = True
        if fast[k]:
            mark["fast"] = True
        if slow[k]:
            mark["slow"] = True
        st = pros[k].get("st")
        before = recent.setdefault(spk[k], [])
        if st is not None:
            if len(before) >= 2:
                base = sorted(before)[len(before) // 2]
                if st - base >= limits.pitch_st:
                    mark["up"] = True
                elif base - st >= limits.pitch_st:
                    mark["down"] = True
            before.append(st)
            del before[:-limits.pitch_context]
        start, end = word.get("start_ms"), word.get("end_ms")
        rate = pros[k].get("rate")
        if rate is not None and rate <= limits.stretch_rate and start is not None and end is not None \
                and end - start >= limits.stretch_ms and not slow[k]:
            mark["stretch"] = "long" if end - start >= limits.long_stretch_ms else True
        out.append(mark)
    return out


def mark_counts(words: list[dict], limits: Limits | None = None) -> dict:
    """How many words each mark falls on (for choosing the limits)."""
    counts = {"words": len(words), "measured": sum(1 for w in words if w.get("pros"))}
    for mark in marks(words, limits):
        for key in mark:
            counts[key] = counts.get(key, 0) + 1
    return counts


def render(words: list[dict], threshold: float, speakers: dict | None = None, limits: Limits | None = None,
           redactions: list[dict] | None = None) -> list[str]:
    """The lines of the view, one per turn, each with its time."""
    limits = limits or Limits()
    speakers = speakers or {}
    words = notation.shown(words)
    word_marks = marks(words, limits)
    loops = notation.repetition_runs(words)
    pending = sorted((r for r in redactions or [] if isinstance(r, dict)), key=lambda r: r["start_ms"])
    ends = [w.get("end_ms") or 0 for w in words[-1:]] + [r.get("end_ms") or 0 for r in redactions or []]
    long_form = max(ends, default=0) >= 3_600_000
    lines: list[str] = []
    line: list[str] = []
    head = ""
    current, previous_end, open_wrap = None, None, None  # open_wrap: "fast" / "slow" / "quiet"
    wrap = {"fast": (">", "<"), "slow": ("<", ">"), "quiet": ("°", "°")}

    def close_wrap() -> None:
        nonlocal open_wrap
        if open_wrap and line:
            line[-1] += wrap[open_wrap][1]
        open_wrap = None

    def flush(latched: bool = False) -> None:
        nonlocal line
        close_wrap()
        if line:
            lines.append(head + " ".join(line) + ("=" if latched else ""))
        line = []

    def label_of(word: dict) -> str | None:
        spk = word.get("speaker")
        return str(spk) if spk not in (None, "") and str(spk) in speakers else None

    i, n = 0, len(words)
    while i < n:
        word = words[i]
        start = word.get("start_ms")
        while pending and start is not None and start >= pending[0]["start_ms"]:
            flush()
            lines.append(notation.redaction_line(pending.pop(0)))
            current, previous_end = None, None
        gap = start - previous_end if start is not None and previous_end is not None else None
        spk = label_of(word)
        change = spk is not None and spk != current
        if change or (gap is not None and gap >= limits.turn_pause_ms) or not line:
            latched = bool(change and line and gap is not None and gap <= limits.latch_ms)
            flush(latched)
            stamp = _clock(start, long_form) + " " if start is not None else ""
            name = f"{speakers[spk]['label']}: " if spk is not None else ""
            head = stamp + name
            if latched:
                line.append("=")
            elif _pause(gap, limits):
                line.append(_pause(gap, limits))
            current = spk if spk is not None else current
        elif _pause(gap, limits):
            close_wrap()
            line.append(_pause(gap, limits))
        # the word, or the group of words that is rendered as one item
        if i in loops:
            group = words[i: loops[i]]
            body, tail = notation._split_trailing(group[-1]["text"])
            text, j = f"[?: {body} ×{len(group)}]{tail}", loops[i]
        elif word.get("span"):
            span = word["span"]
            group = words[i: i + max(span.get("n", 1), 1)]
            body, tail = notation._split_trailing(" ".join(w["text"] for w in group)) if span.get("n", 1) else ("", "")
            last = group[-1] if group else word
            text = notation._with_inserts({"insert_before": word.get("insert_before", []),
                                           "insert_after": last.get("insert_after", []),
                                           "adj_before": word.get("adj_before"),
                                           "adj_after": last.get("adj_after")},
                                          notation._choice([body] + list(span.get("alts", [])), tail,
                                                           word.get("adj")))
            j = i + max(len(group), 1)
        else:
            group = [word]
            plain = not (word.get("alts") or notation._is_uncertain(word, threshold))
            mark = word_marks[i]
            body = word.get("text", "")
            if plain and body:
                if mark.get("stretch"):
                    body = _stretch(body, mark["stretch"] == "long")
                if mark.get("loud"):
                    body = body.upper()
                if mark.get("up"):
                    body = "↑" + body
                elif mark.get("down"):
                    body = "↓" + body
                text = notation._with_inserts(word, body)
            else:
                text = notation._with_inserts(word, notation.render_word(word, threshold))
            j = i + 1
        mark = word_marks[i]
        wanted = next((key for key in ("fast", "slow", "quiet") if mark.get(key)), None)
        if wanted != open_wrap:
            close_wrap()
            if wanted:
                text = wrap[wanted][0] + text
                open_wrap = wanted
        line.append(text)
        timed = [w.get("end_ms") for w in group if w.get("end_ms") is not None]
        previous_end = timed[-1] if timed else previous_end
        i = j
    flush()
    for redaction in pending:
        lines.append(notation.redaction_line(redaction))
    return lines
