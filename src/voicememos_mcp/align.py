"""Align a second reader's text with the primary (timed, confidence-scored) words.

The primary engine's words stay the backbone: they carry timings and measured
confidence. Wherever the second reader disagrees, the primary words get an
alternative reading with ``basis: reader_disagreement``:

* one word replaced by one word    -> ``word["alts"] = ["other"]``           -> ``{a|b}``
* several words replaced / missing -> ``word["span"] = {"n": k, "alts": [...]}`` on the first word
                                      and ``word["in_span"] = True`` on the rest -> ``{a1 a2|b1}``
* extra words only in the reader    -> ``word["insert_after"] = ["b1 b2"]``     -> ``{∅|b1 b2}``

``∅`` means "this reader heard nothing here". Comparison ignores case and
punctuation. The alignment is a plain sequence diff (difflib), which is good
enough to find disagreements but not to decide who is right.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

STRIP = ".,!?;:…\"'“”‘’«»()[]{}-–—"
UNSURE = re.compile(r"^\(\?\)$")


def normalise(word: str) -> str:
    return word.casefold().strip(STRIP).replace("’", "'")


def reader_words(text: str) -> list[dict]:
    """Split a reader's plain text into words, keeping its own "(?)" and "[?]" marks as flags."""
    out: list[dict] = []
    for raw in text.split():
        if UNSURE.match(raw):
            if out:
                out[-1]["unsure"] = True
            continue
        if raw.startswith("[?") and raw.endswith("]"):
            out.append({"text": "", "gap": True})
            continue
        out.append({"text": raw})
    return out


def dominant_language(words: list[dict], min_share: float = 0.8) -> str:
    """Most common word language if it covers ``min_share`` of words, else "mixed"."""
    langs = Counter(w.get("lang") for w in words if w.get("lang"))
    if not langs:
        return "unknown"
    lang, count = langs.most_common(1)[0]
    return lang if count / sum(langs.values()) >= min_share else "mixed"


def align(primary: list[dict], reader_text: str, threshold: float = 0.7) -> tuple[list[dict], dict]:
    """Return primary words annotated with the reader's alternatives, plus agreement metrics."""
    words = [{k: v for k, v in w.items() if k not in ("alts", "span", "in_span", "insert_after", "basis",
                                                      "flag", "reader_unsure")} for w in primary]
    other = reader_words(reader_text)
    a = [normalise(w["text"]) for w in words]
    b = [normalise(w["text"]) for w in other]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)

    matched = substituted = deleted = inserted = spans = spans_low_conf = 0
    low_conf_agreed = low_conf_total = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            for offset in range(i2 - i1):
                if other[j1 + offset].get("unsure"):
                    words[i1 + offset]["reader_unsure"] = True
                conf = words[i1 + offset].get("conf")
                if conf is not None and conf < threshold:
                    low_conf_agreed += 1
            continue
        spans += 1
        alt_text = " ".join(w["text"] for w in other[j1:j2])
        if any((w.get("conf") is not None and w["conf"] < threshold) for w in words[i1:i2]):
            spans_low_conf += 1
        if tag == "insert":
            inserted += j2 - j1
            anchor = max(i1 - 1, 0)
            if words:
                words[anchor].setdefault("insert_after" if i1 > 0 else "insert_before", []).append(alt_text)
                words[anchor]["basis"] = "reader_disagreement"
            continue
        if tag == "delete":
            deleted += i2 - i1
        else:
            substituted += min(i2 - i1, j2 - j1)
            deleted += max(0, (i2 - i1) - (j2 - j1))
            inserted += max(0, (j2 - j1) - (i2 - i1))
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1 and alt_text:
            words[i1]["alts"] = [alt_text]
        else:
            words[i1]["span"] = {"n": i2 - i1, "alts": [alt_text]}
            for k in range(i1 + 1, i2):
                words[k]["in_span"] = True
        words[i1]["basis"] = "reader_disagreement"

    for w in words:
        conf = w.get("conf")
        if conf is not None and conf < threshold:
            low_conf_total += 1
    n_a, n_b = len(a), len(b)
    metrics = {
        "primary_words": n_a,
        "reader_words": n_b,
        "matched": matched,
        "substituted": substituted,
        "deleted": deleted,
        "inserted": inserted,
        "disagreement_spans": spans,
        "agreement": round(matched / max(n_a, n_b), 3) if max(n_a, n_b) else 1.0,
        "word_difference_rate": round((substituted + deleted + inserted) / n_a, 3) if n_a else None,
        "spans_with_low_primary_confidence": spans_low_conf,
        "low_confidence_words": low_conf_total,
        "low_confidence_words_confirmed_by_reader": low_conf_agreed,
        "reader_unsure_marks": sum(1 for w in other if w.get("unsure")),
        "reader_gaps": sum(1 for w in other if w.get("gap")),
    }
    return words, metrics
