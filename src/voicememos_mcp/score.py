"""Score readers against a listening checklist the user has marked.

The second-reader report (``_reports/second-reader_*.md``) lists spots where the primary
engine (A) and the second reader (B) disagree, and asks the listener to mark the better
reading (``better: A`` / ``B`` / ``both`` / ``neither``). Once marked, the file is ground
truth for those spots, and any other reading of the same memo can be scored against it:
does the reading say what the listener heard?

A spot is found in a reading by its context (the words around it), not by position, so
readings from any model can be scored, and so can a primary transcript made again later.
Words are compared as written, apart from case and punctuation: spoken forms count
("nään" is not "näen"), since a transcript is meant to say what was said.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

TOKEN = re.compile(r"\w+(?:[:.\-'’]\w+)*")
EMPTY = "∅"
MAX_SPAN = 14  # words between the two contexts


def tokens(text: str) -> list[str]:
    return [t.casefold().replace("’", "'") for t in TOKEN.findall(text or "")]


@dataclass
class Spot:
    memo: str
    at: str
    a: str
    b: str
    left: list[str]
    right: list[str]
    accept: set[str]  # readings the listener accepts: "A", "B", or both
    label: str
    lang: str = ""

    def reading(self, which: str) -> list[str]:
        text = self.a if which == "A" else self.b
        return [] if text.strip() == EMPTY else tokens(text)


@dataclass
class Verdict:
    spot: Spot
    said: list[str] | None  # None: the spot could not be found in the reading
    matches: str = ""  # "A", "B", "A+B" (A and B read the same) or "" (neither)
    located: float = 0.0

    @property
    def correct(self) -> bool | None:
        if self.said is None:
            return None
        return any(m in self.spot.accept for m in self.matches.split("+") if m)


@dataclass
class Score:
    reader: str
    verdicts: list[Verdict] = field(default_factory=list)

    def counts(self, lang: str | None = None) -> dict:
        vs = [v for v in self.verdicts if lang is None or v.spot.lang == lang]
        return {"spots": len(vs), "correct": sum(v.correct is True for v in vs),
                "wrong": sum(v.correct is False for v in vs), "not_found": sum(v.correct is None for v in vs)}


def _accept(answer: str) -> set[str] | None:
    """Readings the listener accepts, from the ``better:`` line; None when unmarked or unusable."""
    text = answer.strip().casefold()
    if not text or text.startswith("a / b") or "/" in text.split("(")[0]:
        return None
    head = re.split(r"[\s(,]", text, maxsplit=1)[0]
    extra = text[len(head):]
    if head in ("both", "kumpikin", "molemmat") or (head in ("a", "b") and "both" in extra.split(":")[0]):
        return {"A", "B"}
    if head in ("a", "b"):
        return {head.upper()}
    return None  # neither, ? or free text: nothing to score against


def parse_labels(markdown: str) -> list[Spot]:
    """Marked spots from a second-reader report (both the flat and the grouped layout)."""
    spots: list[Spot] = []
    lang = memo = ""
    current: dict | None = None

    def flush():
        if current and current.get("better") is not None:
            accept = _accept(current["better"])
            if accept and "context" in current:
                left, _, right = current["context"].partition("▮")
                spots.append(Spot(current["memo"], current["at"], current.get("a", EMPTY), current.get("b", EMPTY),
                                  tokens(left), tokens(right), accept, current["better"].strip(), current["lang"]))

    for line in markdown.splitlines():
        if line.startswith("## Samples:"):
            lang = "en" if "English" in line else "fi" if "Finnish" in line else line.split(":")[1].split()[0]
            continue
        heading = re.match(r"###\s.*`([0-9a-f]{8})`", line)
        if heading:
            memo = heading.group(1)
            continue
        item = re.match(r"- \[[ xX]\] (?:\*\*[^*]*\*\*\s.*?`([0-9a-f]{8})`\s)?at ([0-9:]+)", line)
        if item:
            flush()
            current = {"memo": item.group(1) or memo, "at": item.group(2), "lang": lang}
            continue
        field_ = re.match(r"\s+- (A \(primary\)|B \(reader\)|context|better):\s?(.*)$", line)
        if field_ and current is not None:
            key = {"A (primary)": "a", "B (reader)": "b"}.get(field_.group(1), field_.group(1))
            current[key] = field_.group(2)
    flush()
    return spots


def _ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def locate(spot: Spot, reading: list[str]) -> tuple[list[str] | None, float]:
    """The words of ``reading`` between the spot's left and right context, and how well they matched."""
    n = len(reading)
    k_left, k_right = len(spot.left), len(spot.right)
    best: tuple[float, int, int] = (-1.0, 0, 0)
    ends = [0] if not spot.left else range(1, n + 1)
    for e in ends:
        left_score = 1.0 if not spot.left else _ratio(spot.left, reading[max(0, e - k_left):e])
        if left_score < 0.4:
            continue
        starts = [n] if not spot.right else range(e, min(n, e + MAX_SPAN) + 1)
        for s in starts:
            right_score = 1.0 if not spot.right else _ratio(spot.right, reading[s:s + k_right])
            score = (left_score + right_score) / 2
            # prefer the shorter span on ties: the context words belong to the context
            if score > best[0] + 1e-9 or (abs(score - best[0]) <= 1e-9 and s - e < best[2] - best[1]):
                best = (score, e, s)
    score, e, s = best
    if score < 0.6:
        return None, max(score, 0.0)
    return reading[e:s], score


def judge(spot: Spot, reading: list[str]) -> Verdict:
    said, located = locate(spot, reading)
    if said is None:
        return Verdict(spot, None, located=located)
    matches = "+".join(w for w in ("A", "B") if said == spot.reading(w))
    return Verdict(spot, said, matches, located)


def readings(data: dict) -> dict[str, list[str]]:
    """Every reading in a transcript: the primary words and each reader's text, by name."""
    out = {f"soniox ({(data.get('engine') or {}).get('model', 'primary')})":
           tokens(" ".join(w.get("text", "") for w in data.get("words", [])))}
    for reader in data.get("readers", []):
        if reader.get("role") not in ("second", "candidate"):
            continue
        text = reader.get("text")
        if text is None and reader.get("chunks"):
            text = " ".join(c.get("text") or "" for c in reader["chunks"])
        if text is None and reader.get("error", "").startswith("rejected"):
            text = reader.get("text") or ""
        if text is None:
            continue
        name = reader.get("requested_model") or reader.get("model") or "reader"
        if reader.get("role") == "second":
            name += " (active)"
        out[name] = tokens(text)
    return out


def score(spots: list[Spot], transcripts: dict[str, dict]) -> dict[str, Score]:
    """Score every reading found in ``transcripts`` (short memo id -> transcript JSON)."""
    scores: dict[str, Score] = {}
    for spot in spots:
        data = transcripts.get(spot.memo)
        if data is None:
            continue
        for name, words in readings(data).items():
            scores.setdefault(name, Score(name)).verdicts.append(judge(spot, words))
    return scores


def render(scores: dict[str, Score]) -> str:
    langs = sorted({v.spot.lang for s in scores.values() for v in s.verdicts})
    lines = ["| Reader | " + " | ".join(f"{lang} correct" for lang in langs) + " | all | not found |",
             "|---|" + "---|" * (len(langs) + 2)]
    for name, sc in sorted(scores.items()):
        cells = []
        for lang in langs:
            c = sc.counts(lang)
            cells.append(f"{c['correct']}/{c['spots'] - c['not_found']}")
        total = sc.counts()
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {total['correct']}/{total['spots'] - total['not_found']} | {total['not_found']} |")
    lines += ["", "Wrong or not found (what the reader has at the spot):", ""]
    for name, sc in sorted(scores.items()):
        for v in sc.verdicts:
            if v.correct is not True:
                said = "(not found)" if v.said is None else (" ".join(v.said) or EMPTY)
                lines.append(f"- {name} · {v.spot.memo} {v.spot.at} · heard {'/'.join(sorted(v.spot.accept))} "
                             f"(A: {v.spot.a} | B: {v.spot.b}) · reader: {said}")
    return "\n".join(lines)
