"""A third listening of the spots where the two readers disagree.

Where the primary engine and the second reader wrote different things, the transcript shows
both (``{a|b}``) and nobody knows who is right. Here a model hears a few seconds around one
such spot, is given both readings without being told whose they are (their order is shuffled
per spot), and says which one it hears, or something else, or that it cannot tell. The answer
is stored on the word and shown as a ``*`` on the reading it sided with: ``{a*|b}``, or
``{a|b|c*}`` when it heard a third thing. Nothing is replaced: both readings stay.

It is a third opinion, not a verdict. A model of the same family as the second reader tends to
hear what its sibling heard, and a listener who is shown two readings leans towards them, so
the record keeps how often each side was picked (``readers[role=adjudicator]``).

One request per spot, most doubtful spots first (lowest confidence of the primary engine),
at most ``VOICEMEMOS_ADJUDICATE_MAX`` per recording.
"""

from __future__ import annotations

import json
import re

from . import notation

CLIP_MARGIN_S = 3.0
CONTEXT_WORDS = 10
NOTHING = "(nothing)"

PROMPT = """You hear a short clip cut from a longer recording. Two transcribers disagree about one spot in it.

Language of the recording: {languages}.

Words before the spot: "{before}"
Reading A: "{a}"
Reading B: "{b}"
Words after the spot: "{after}"

"{nothing}" means that transcriber heard no words at the spot. The words before and after are there to help you \
find the spot; they may contain mistakes too.

Listen to the clip and decide what is said at the spot. Answer with JSON only:
{{"pick": "A" | "B" | "other" | "unclear", "heard": "<the words you hear at the spot, verbatim, in the spoken language>"}}

Pick "other" only if you clearly hear something that is neither reading. Pick "unclear" if you cannot tell."""


def _plain(words: list[dict]) -> str:
    return " ".join(w.get("text", "") for w in words if w.get("text"))


def _swapped(index: int, kind: str) -> bool:
    """Whether the readings change places in the request: fixed per spot, about half of them."""
    return (index * 2654435761 + len(kind) * 40503) % 1000 < 500


def disputes(words: list[dict]) -> list[dict]:
    """Every spot where the readers disagree: ``{"index", "kind", "a", "b", "start_ms", "end_ms",
    "before", "after", "conf"}``. ``a`` is the primary engine's reading and ``b`` the second
    reader's ("" = heard nothing); ``kind`` is alts, span, before or after (the last two: words
    only the second reader heard, before or after word ``index``)."""
    out = []
    n = len(words)

    def context(first: int, last: int) -> tuple[str, str]:
        return _plain(words[max(first - CONTEXT_WORDS, 0):first]), _plain(words[last:last + CONTEXT_WORDS])

    def add(index: int, kind: str, a: str, b: str, first: int, last: int, start, end) -> None:
        if start is None or end is None:
            return
        before, after = context(first, last)
        confs = [w["conf"] for w in words[first:last] if w.get("conf") is not None]
        out.append({"index": index, "kind": kind, "a": a, "b": b, "start_ms": start, "end_ms": max(end, start),
                    "before": before, "after": after, "conf": min(confs) if confs else None})

    for i, word in enumerate(words):
        if word.get("quarantine"):
            continue
        if word.get("alts"):
            add(i, "alts", notation._split_trailing(word.get("text", ""))[0], str(word["alts"][0]), i, i + 1,
                word.get("start_ms"), word.get("end_ms"))
        elif word.get("span"):
            span = word["span"]
            last = min(i + max(span.get("n", 1), 1), n)
            group = words[i:last]
            ends = [w.get("end_ms") for w in group if w.get("end_ms") is not None]
            add(i, "span", notation._split_trailing(_plain(group))[0] if span.get("n", 1) else "",
                str((span.get("alts") or [""])[0]), i, last, word.get("start_ms"), ends[-1] if ends else None)
        if word.get("insert_before"):
            add(i, "before", "", str(word["insert_before"][0]), i, i, max((word.get("start_ms") or 0) - 1500, 0),
                word.get("start_ms"))
        if word.get("insert_after"):
            nxt = words[i + 1].get("start_ms") if i + 1 < n else None
            end = word.get("end_ms")
            add(i, "after", "", str(word["insert_after"][0]), i + 1, i + 1, end,
                nxt if nxt is not None and end is not None and nxt > end else (end + 1500 if end is not None else None))
    return out


def select(found: list[dict], limit: int) -> list[dict]:
    """At most ``limit`` spots, the most doubtful first (lowest primary confidence; words only the
    second reader heard come last), returned in the order of the recording."""
    if limit and len(found) > limit:
        ranked = sorted(found, key=lambda d: (d["conf"] is None, d["conf"] if d["conf"] is not None else 1.0))
        found = ranked[:limit]
    return sorted(found, key=lambda d: (d["start_ms"], d["index"]))


def build_prompt(dispute: dict, languages: list[str]) -> str:
    from .gemini import LANGUAGE_NAMES

    a, b = dispute["a"] or NOTHING, dispute["b"] or NOTHING
    if _swapped(dispute["index"], dispute["kind"]):
        a, b = b, a
    names = ", ".join(LANGUAGE_NAMES.get(code, code) for code in languages) or "unknown"
    return PROMPT.format(languages=names, before=dispute["before"], after=dispute["after"], a=a, b=b,
                         nothing=NOTHING)


def clip_bounds(dispute: dict) -> tuple[float, float]:
    return (max(dispute["start_ms"] / 1000 - CLIP_MARGIN_S, 0.0), dispute["end_ms"] / 1000 + CLIP_MARGIN_S)


def parse_answer(text: str, dispute: dict) -> dict | None:
    """The model's answer as stored on the word: ``{"pick": 0 | 1 | "other" | "unclear", "heard"}``
    (0 = the primary engine's reading, 1 = the second reader's). None when it cannot be read."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        answer = json.loads(match.group(0)) if match else None
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        return None
    pick = str(answer.get("pick", "")).strip().lower()
    heard = str(answer.get("heard") or "").strip()[:200]
    if heard.casefold() == NOTHING:
        heard = ""
    if pick in ("a", "b"):
        index = 0 if pick == "a" else 1
        if _swapped(dispute["index"], dispute["kind"]):
            index = 1 - index
        return {"pick": index, **({"heard": heard} if heard else {})}
    if pick == "other" and heard:
        return {"pick": "other", "heard": heard}
    if pick in ("unclear", "other"):
        return {"pick": "unclear"}
    return None


FIELD = {"alts": "adj", "span": "adj", "before": "adj_before", "after": "adj_after"}


def clear(words: list[dict]) -> list[dict]:
    return [{k: v for k, v in w.items() if k not in ("adj", "adj_before", "adj_after")} for w in words]


def apply(words: list[dict], dispute: dict, answer: dict) -> None:
    words[dispute["index"]][FIELD[dispute["kind"]]] = answer


def tally(words: list[dict]) -> dict:
    """How the third listening went: spots judged and which side it took."""
    counts = {"judged": 0, "primary": 0, "second": 0, "other": 0, "unclear": 0}
    for word in words:
        for field in ("adj", "adj_before", "adj_after"):
            answer = word.get(field)
            if not isinstance(answer, dict):
                continue
            counts["judged"] += 1
            pick = answer.get("pick")
            counts["primary" if pick == 0 else "second" if pick == 1 else "other" if pick == "other" else "unclear"] += 1
    return counts
