"""Align a second reader's text with the primary (timed, confidence-scored) words.

The primary engine's words stay the backbone: they carry timings and measured
confidence. Wherever the second reader disagrees, the primary words get an
alternative reading with ``basis: reader_disagreement``:

* one word replaced by one word    -> ``word["alts"] = ["other"]``           -> ``{a|b}``
* several words replaced / missing -> ``word["span"] = {"n": k, "alts": [...]}`` on the first word
                                      and ``word["in_span"] = True`` on the rest -> ``{a1 a2|b1}``
* extra words only in the reader    -> ``word["insert_after"] = ["b1 b2"]``     -> ``{∅|b1 b2}``

``∅`` means "this reader heard nothing here". Comparison ignores case and
punctuation, and in Finnish also the difference between spoken and written forms
of the same word ("mä"/"minä", "oon"/"olen"): one reader often writes what was
said and the other what was meant, which is not a hearing disagreement. The alignment is a plain sequence diff (difflib), which is good
enough to find disagreements but not to decide who is right.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

STRIP = ".,!?;:…\"'“”‘’«»()[]{}-–—"
UNSURE = re.compile(r"^\(\?\)$")


# Spoken Finnish -> written form. Only whole words with one obvious written form;
# no suffix rules, because "huomasin"/"huomasi" is a real difference.
SPOKEN_FI = {
    "mä": "minä", "mie": "minä", "sä": "sinä", "sie": "sinä", "mun": "minun", "sun": "sinun",
    "meiän": "meidän", "teiän": "teidän", "oon": "olen", "oot": "olet", "ootte": "olette", "oo": "ole",
    "tää": "tämä", "toi": "tuo", "noi": "nuo", "nää": "nämä",
    "sit": "sitten", "sitte": "sitten", "niinku": "niinkuin", "niinkun": "niinkuin", "ninku": "niinkuin",
    "kans": "kanssa", "vaik": "vaikka", "mut": "mutta", "miks": "miksi", "jotain": "jotakin",
    "tääl": "täällä", "siel": "siellä", "tarvi": "tarvitse", "tarvii": "tarvitse", "vähä": "vähän",
    "tiiä": "tiedä", "tiiän": "tiedän", "ollu": "ollut", "tullu": "tullut", "menny": "mennyt",
    "tehny": "tehnyt", "semmonen": "sellainen", "tämmönen": "tällainen", "tommonen": "tuollainen",
    "ku": "kun",
}
# Spoken forms that are also common English words: folded only in Finnish context.
AMBIGUOUS_EN = {"sun", "sit", "mut"}


def normalise(word: str) -> str:
    return word.casefold().strip(STRIP).replace("’", "'")


def comparison_key(word: str, lang: str | None) -> str:
    """Normalised word, with Finnish spoken forms folded to the written form when the word is Finnish."""
    key = normalise(word)
    if key in SPOKEN_FI and (lang == "fi" or (lang is None and key not in AMBIGUOUS_EN)):
        return SPOKEN_FI[key]
    return key


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


READER_FIELDS = ("alts", "span", "in_span", "insert_after", "insert_before", "basis", "flag", "reader_unsure")


def strip_reader(primary: list[dict]) -> list[dict]:
    """Primary words without any second-reader annotations."""
    return [{k: v for k, v in w.items() if k not in READER_FIELDS} for w in primary]


def align(primary: list[dict], reader_text: str, threshold: float = 0.7) -> tuple[list[dict], dict]:
    """Return primary words annotated with the reader's alternatives, plus agreement metrics."""
    words = strip_reader(primary)
    other = reader_words(reader_text)
    memo_lang = dominant_language(words)
    reader_lang = "fi" if memo_lang == "fi" else None  # mixed memos: fold only the unambiguous forms
    a = [comparison_key(w["text"], w.get("lang") or reader_lang) for w in words]
    b = [comparison_key(w["text"], reader_lang) for w in other]
    folded = sum(1 for w, k in zip(words, a) if normalise(w["text"]) != k) \
        + sum(1 for w, k in zip(other, b) if normalise(w["text"]) != k)
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
        "spoken_forms_folded": folded,
    }
    return words, metrics


# ----------------------------------------------------------------------------- long recordings
#
# A long recording is read by the second reader in parts of about equal length
# (default about 30 min). Cuts fall in a pause between two words, preferably after
# a sentence, never inside a word. Each part is aligned with the primary words of
# the same span; the transcript itself stays one piece.

SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
COUNT_KEYS = ("primary_words", "reader_words", "matched", "substituted", "deleted", "inserted",
              "disagreement_spans", "spans_with_low_primary_confidence", "low_confidence_words",
              "low_confidence_words_confirmed_by_reader", "reader_unsure_marks", "reader_gaps",
              "spoken_forms_folded")


def plan_chunks(words: list[dict], chunk_ms: int, window_ms: int = 5 * 60_000) -> list[dict]:
    """Split the word list into parts of about ``chunk_ms``.

    Returns ``[{"first", "last", "start_ms", "end_ms"}]``: word index range (``last``
    exclusive) and audio span (``end_ms`` None for the last part).
    """
    timed = [i for i, w in enumerate(words) if w.get("start_ms") is not None and w.get("end_ms") is not None]
    if not timed or chunk_ms <= 0:
        return [{"first": 0, "last": len(words), "start_ms": 0, "end_ms": None}]
    total = words[timed[-1]]["end_ms"]
    parts = max(1, round(total / chunk_ms))
    length = total / parts
    gaps = []  # (cut_ms, index of first word after the cut, pause length, after a sentence)
    for a, b in zip(timed, timed[1:]):
        end, start = words[a]["end_ms"], words[b]["start_ms"]
        gaps.append(((end + max(start, end)) // 2, b, max(start - end, 0),
                     bool(SENTENCE_END.search(words[a].get("text", "")))))
    cuts = []
    for k in range(1, parts):
        target = k * length
        near = [g for g in gaps if abs(g[0] - target) <= window_ms and (not cuts or g[1] > cuts[-1][1])]
        if not near:
            near = [g for g in gaps if not cuts or g[1] > cuts[-1][1]]
        if not near:
            break
        best = max(near, key=lambda g: (g[3], g[2] >= 400, -abs(g[0] - target) // 60_000, g[2]))
        cuts.append(best)
    out, first, start_ms = [], 0, 0
    for cut_ms, index, _, _ in cuts:
        out.append({"first": first, "last": index, "start_ms": start_ms, "end_ms": cut_ms})
        first, start_ms = index, cut_ms
    out.append({"first": first, "last": len(words), "start_ms": start_ms, "end_ms": None})
    return out


def align_chunks(primary: list[dict], chunks: list[dict], threshold: float = 0.7) -> tuple[list[dict], dict]:
    """Align each part that has reader text with its own span of the primary words."""
    words: list[dict] = []
    totals = dict.fromkeys(COUNT_KEYS, 0)
    read = 0
    for chunk in chunks:
        part = primary[chunk["first"]:chunk["last"]]
        if chunk.get("text") is None:
            words += part
            continue
        aligned, metrics = align(part, chunk["text"], threshold)
        words += aligned
        read += 1
        for key in COUNT_KEYS:
            totals[key] += metrics.get(key, 0)
    n_a, n_b = totals["primary_words"], totals["reader_words"]
    totals.update(
        agreement=round(totals["matched"] / max(n_a, n_b), 3) if max(n_a, n_b) else 1.0,
        word_difference_rate=round((totals["substituted"] + totals["deleted"] + totals["inserted"]) / n_a, 3)
        if n_a else None,
        chunks_read=read, chunks_total=len(chunks),
    )
    return words, totals
