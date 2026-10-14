"""Words, uncertainty and the inline notation.

Two layers are kept apart:

* **Canonical data (JSON).** One entry per word with ``conf`` (0..1 or null),
  optional ranked ``alts`` and a ``basis`` saying where the uncertainty comes
  from. The semantics follow PAGE XML (``TextEquiv@conf``, alternatives ranked by
  ``index``) so the data can be exported to PAGE/ALTO/TEI later.

* **Inline notation (Markdown).** Rendered from the data on demand, so the
  threshold can change without re-transcribing:

  ===================  ==========================================================
  ``word (?)``         best guess, low confidence
  ``[?: w1 w2 w3]``    run of consecutive low-confidence words (best guesses kept)
  ``{a|b}``            two plausible readings, more likely first (multi-engine)
  ``[?]``              something was said but nothing usable was recognised
  ===================  ==========================================================

  This mirrors conversation-analysis (Jefferson) practice: ``(word)`` uncertain
  hearing, ``(a)/(b)`` alternative hearings, ``( )`` unhearable.

Basis values:

* ``asr_confidence``  - measured per-token confidence from the speech engine
* ``reader_disagreement`` - independent engines disagreed
* ``self_report``     - a model said it was unsure (not calibrated)
"""

from __future__ import annotations

import re

PUNCTUATION = ".,!?;:…-–—\"'()"
TRAILING_PUNCTUATION = ".,!?;:…"

READER_NOTE = (
    "Reading guide: `word (?)` and `[?: …]` are the speech engine's best guesses with low confidence; "
    "`{a|b}` lists two plausible readings, more likely first; `[?]` marks speech that was not recognised. "
    "Use marked words, but do not present them as certain. If context suggests a different word, propose "
    "it visibly as `word→suggestion (?)` instead of replacing it silently."
)


# ----------------------------------------------------------------------------- tokens -> words

def _is_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(ch in PUNCTUATION for ch in stripped)


def tokens_to_words(tokens: list[dict]) -> tuple[list[dict], bool]:
    """Join Soniox sub-word tokens into words.

    A token whose text starts with whitespace begins a new word. A word's
    confidence is the minimum over its letter-bearing tokens (punctuation does not
    lower it). Returns ``(words, had_confidence)``.
    """
    words: list[dict] = []
    had_confidence = False
    for token in tokens:
        text = token.get("text", "")
        if not text or token.get("translation_status") == "translation":
            continue
        stripped = text.strip()
        if stripped.startswith("<") and stripped.endswith(">"):  # control tokens such as <end>
            continue
        conf = token.get("confidence")
        if conf is not None:
            had_confidence = True
        punctuation = _is_punctuation(text)
        new_word = not words or (text[0].isspace() and not punctuation)
        if new_word:
            words.append({
                "text": stripped,
                "start_ms": token.get("start_ms"),
                "end_ms": token.get("end_ms"),
                "conf": None if (conf is None or punctuation) else float(conf),
                "lang": token.get("language"),
            })
        else:
            word = words[-1]
            word["text"] += stripped if text[0].isspace() else text
            if conf is not None and not punctuation:
                word["conf"] = float(conf) if word["conf"] is None else min(word["conf"], float(conf))
            if token.get("end_ms") is not None:
                word["end_ms"] = token["end_ms"]
            if not word.get("lang") and token.get("language"):
                word["lang"] = token["language"]
    return [w for w in words if w["text"]], had_confidence


def annotate(words: list[dict], threshold: float) -> list[dict]:
    """Add ``flag``/``basis`` to each word for the given threshold (returns new dicts)."""
    out = []
    for word in words:
        item = dict(word)
        conf = item.get("conf")
        if item.get("alts"):
            item["flag"] = "alternatives"
            item.setdefault("basis", "reader_disagreement")
        elif conf is not None and conf < threshold:
            item["flag"] = "uncertain"
            item["basis"] = "asr_confidence"
        else:
            item.pop("flag", None)
        out.append(item)
    return out


# ----------------------------------------------------------------------------- rendering

def _split_trailing(word: str) -> tuple[str, str]:
    end = len(word)
    while end > 0 and word[end - 1] in TRAILING_PUNCTUATION:
        end -= 1
    return word[:end], word[end:]


def _is_uncertain(word: dict, threshold: float) -> bool:
    conf = word.get("conf")
    return conf is not None and conf < threshold and not word.get("alts")


def render_word(word: dict, threshold: float) -> str:
    text = word.get("text", "")
    if not text:
        return "[?]"
    if word.get("alts"):
        body, tail = _split_trailing(text)
        options = [body] + [_split_trailing(a)[0] for a in word["alts"]]
        return "{" + "|".join(dict.fromkeys(o for o in options if o)) + "}" + tail
    if _is_uncertain(word, threshold):
        body, tail = _split_trailing(text)
        return f"{body} (?){tail}" if body else tail
    return text


def render_blocks(words: list[dict], threshold: float, run_min: int = 4, pause_ms: int = 2500) -> list[str]:
    """Render words to paragraphs. A pause longer than ``pause_ms`` starts a new paragraph."""
    blocks: list[list[str]] = [[]]
    i, n = 0, len(words)
    while i < n:
        word = words[i]
        if i > 0 and blocks[-1]:
            prev_end, start = words[i - 1].get("end_ms"), word.get("start_ms")
            if prev_end is not None and start is not None and start - prev_end > pause_ms:
                blocks.append([])
        if _is_uncertain(word, threshold):
            j = i
            while j < n and _is_uncertain(words[j], threshold):
                j += 1
            run = words[i:j]
            if len(run) >= run_min:
                body, tail = _split_trailing(" ".join(w["text"] for w in run))
                blocks[-1].append(f"[?: {body}]{tail}")
            else:
                blocks[-1].extend(render_word(w, threshold) for w in run)
            i = j
            continue
        blocks[-1].append(render_word(word, threshold))
        i += 1
    return [" ".join(b) for b in blocks if b]


def stats(words: list[dict], threshold: float) -> dict:
    """Counts and an overall certainty level (share of marked words)."""
    total = len(words)
    uncertain = sum(1 for w in words if _is_uncertain(w, threshold))
    alternatives = sum(1 for w in words if w.get("alts"))
    share = (uncertain + alternatives) / total if total else 0.0
    if total == 0:
        level = "no speech"
    elif share < 0.05:
        level = "high"
    elif share < 0.15:
        level = "moderate"
    else:
        level = "low"
    return {
        "words": total,
        "uncertain": uncertain,
        "alternatives": alternatives,
        "marked_share": round(share, 3),
        "certainty": level,
        "threshold": threshold,
    }


def stats_line(s: dict) -> str:
    if not s["words"]:
        return "no speech recognised"
    return (f"{s['words']} words, {s['uncertain'] + s['alternatives']} marked "
            f"({100 * s['marked_share']:.0f} %), certainty {s['certainty']}")


# ----------------------------------------------------------------------------- spoken markers

def extract_markers(text: str, markers: dict[str, str]) -> list[dict]:
    """Find sentences that start with a spoken marker word, e.g. "Idea: ...".

    ``markers`` maps the spoken word (lower case) to the label shown in notes.
    """
    if not markers or not text:
        return []
    words = "|".join(sorted((re.escape(k) for k in markers), key=len, reverse=True))
    pattern = re.compile(
        rf"(?:^|(?<=[.!?…\n]))\s*(?P<kw>{words})\b[\s:,.\-–]*"
        rf"(?P<body>(?:\[[^\]]*\]|\{{[^}}]*\}}|\(\?\)|[^.!?…\[{{\n])+[.!?…]?)",
        re.IGNORECASE,
    )
    found = []
    for match in pattern.finditer(text):
        body = match.group("body").strip().rstrip(".!?… ")
        if len(body) >= 3:
            found.append({"label": markers[match.group("kw").lower()], "text": body[:400]})
    return found
