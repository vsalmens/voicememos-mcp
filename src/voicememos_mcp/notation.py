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
  ``{a|b}``            two engines disagree: primary engine first, second reader after
  ``{a b|∅}``          the second reader heard nothing here (``{∅|b}``: only it heard something)
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
from dataclasses import dataclass, field

PUNCTUATION = ".,!?;:…-–—\"'()"
TRAILING_PUNCTUATION = ".,!?;:…"

READER_NOTE = (
    "Reading guide: `word (?)` and `[?: …]` are the speech engine's best guesses with low confidence; "
    "`{a|b}` means two independent engines heard different things (primary engine first, neither is "
    "known to be right; `∅` = that engine heard nothing); `[?]` marks speech that was not recognised. "
    "Use marked words, but do not present them as certain. If context suggests a different word, propose "
    "it visibly as `word→suggestion (?)` instead of replacing it silently. `[?: word ×N]` means the "
    "engine produced the same word N times in a row, usually a mishearing over music or noise. "
    "`[mm:ss]` / `[h:mm:ss]` at the start of a paragraph is its time in the recording."
)

REPEAT_MIN = 4  # this many identical words in a row are treated as a recognition loop


def _norm_word(text: str) -> str:
    return re.sub(r"[^\w]+", "", (text or "").lower())


def repetition_runs(words: list[dict], min_len: int = REPEAT_MIN) -> dict[int, int]:
    """Start index -> end index (exclusive) of runs of the same word by the same speaker."""
    runs, i, n = {}, 0, len(words)
    while i < n:
        key = _norm_word(words[i].get("text", ""))
        j = i + 1
        while (key and j < n and _norm_word(words[j].get("text", "")) == key
               and words[j].get("speaker") == words[i].get("speaker")
               and not words[j].get("span") and not _has_alternative(words[j])):
            j += 1
        if key and j - i >= min_len and not words[i].get("span") and not _has_alternative(words[i]):
            runs[i] = j
        i = j
    return runs


def _clock(ms: int, long_form: bool) -> str:
    sec = int(ms // 1000)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"[{h}:{m:02d}:{s:02d}]" if long_form else f"[{m:02d}:{s:02d}]"


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
                "speaker": token.get("speaker"),
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


def _has_alternative(word: dict) -> bool:
    return bool(word.get("alts") or word.get("span") or word.get("in_span"))


def annotate(words: list[dict], threshold: float) -> list[dict]:
    """Add ``flag``/``basis`` to each word for the given threshold (returns new dicts)."""
    repeated = {k for a, b in repetition_runs(words).items() for k in range(a, b)}
    out = []
    for idx, word in enumerate(words):
        item = dict(word)
        conf = item.get("conf")
        if _has_alternative(item):
            item["flag"] = "alternatives"
            item.setdefault("basis", "reader_disagreement")
        elif idx in repeated:
            item["flag"] = "repetition"
            item["basis"] = "repetition_loop"
        elif conf is not None and conf < threshold:
            item["flag"] = "uncertain"
            item["basis"] = "asr_confidence"
        else:
            item.pop("flag", None)
            if item.get("basis") in ("asr_confidence", "repetition_loop"):
                item.pop("basis")
        out.append(item)
    return out


# ----------------------------------------------------------------------------- rendering

EMPTY = "∅"


def _split_trailing(word: str) -> tuple[str, str]:
    end = len(word)
    while end > 0 and word[end - 1] in TRAILING_PUNCTUATION:
        end -= 1
    return word[:end], word[end:]


def _is_uncertain(word: dict, threshold: float) -> bool:
    conf = word.get("conf")
    return conf is not None and conf < threshold and not _has_alternative(word)


def _choice(options: list[str], tail: str = "") -> str:
    cleaned = list(dict.fromkeys(_split_trailing(o)[0] if o else EMPTY for o in options))
    return "{" + "|".join(o or EMPTY for o in cleaned) + "}" + tail


def render_word(word: dict, threshold: float) -> str:
    text = word.get("text", "")
    if not text:
        return "[?]"
    if word.get("alts"):
        body, tail = _split_trailing(text)
        return _choice([body] + list(word["alts"]), tail)
    if _is_uncertain(word, threshold):
        body, tail = _split_trailing(text)
        return f"{body} (?){tail}" if body else tail
    return text


def _with_inserts(word: dict, rendered: str) -> str:
    before = " ".join(_choice([""] + [alt]) for alt in word.get("insert_before", []))
    after = " ".join(_choice([""] + [alt]) for alt in word.get("insert_after", []))
    return " ".join(x for x in (before, rendered, after) if x)


def speaker_labels(words: list[dict]) -> dict:
    """Label speakers when more than one was detected; ``{}`` for single-speaker memos.

    Speakers are numbered S1, S2, ... in order of appearance. The labels only hold within
    one memo; nobody is named.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for word in words:
        spk = word.get("speaker")
        if spk in (None, ""):
            continue
        spk = str(spk)
        if spk not in counts:
            order.append(spk)
            counts[spk] = 0
        counts[spk] += 1
    if len(order) < 2:
        return {}
    total = sum(counts.values())
    return {spk: {"label": f"S{number}", "words": counts[spk], "share": round(counts[spk] / total, 3)}
            for number, spk in enumerate(order, start=1)}


def render_blocks(words: list[dict], threshold: float, run_min: int = 4, pause_ms: int = 2500,
                  speakers: dict | None = None, stamp_every_s: int = 0) -> list[str]:
    """Render words to paragraphs.

    A pause longer than ``pause_ms`` starts a new paragraph. With ``speakers`` (from
    ``speaker_labels``), a change of speaker also starts one, prefixed with ``**label:**``.
    With ``stamp_every_s``, a paragraph that starts at least that long after the previous
    time stamp gets its time (``[mm:ss]``, or ``[h:mm:ss]`` in recordings of an hour or more).
    Runs of the same word (``REPEAT_MIN`` or more) are collapsed to ``[?: word ×N]``.
    """
    speakers = speakers or {}
    blocks: list[list[str]] = [[]]
    prefixes: list[str] = [""]
    starts: list[int | None] = [None]
    current = None
    loops = repetition_runs(words)

    def speaker_of(word: dict):
        spk = word.get("speaker")
        return str(spk) if spk not in (None, "") and str(spk) in speakers else None

    i, n = 0, len(words)
    while i < n:
        word = words[i]
        spk = speaker_of(word)
        if spk is not None and spk != current:
            if blocks[-1]:
                blocks.append([])
                prefixes.append("")
                starts.append(None)
            prefixes[-1] = f"**{speakers[spk]['label']}:** "
            current = spk
        elif i > 0 and blocks[-1]:
            prev_end, start = words[i - 1].get("end_ms"), word.get("start_ms")
            if prev_end is not None and start is not None and start - prev_end > pause_ms:
                blocks.append([])
                prefixes.append("")
                starts.append(None)
        if starts[-1] is None and word.get("start_ms") is not None:
            starts[-1] = word["start_ms"]
        if i in loops:
            group = words[i: loops[i]]
            body, tail = _split_trailing(group[-1]["text"])
            blocks[-1].append(f"[?: {body} ×{len(group)}]{tail}")
            i = loops[i]
            continue
        if word.get("span"):
            span = word["span"]
            group = words[i: i + max(span.get("n", 1), 1)]
            body, tail = _split_trailing(" ".join(w["text"] for w in group)) if span.get("n", 1) else ("", "")
            rendered = _choice([body] + list(span.get("alts", [])), tail)
            last = group[-1] if group else word
            blocks[-1].append(_with_inserts({"insert_before": word.get("insert_before", []),
                                             "insert_after": last.get("insert_after", [])}, rendered))
            i += max(len(group), 1)
            continue
        if _is_uncertain(word, threshold) and not word.get("insert_after") and not word.get("insert_before"):
            j = i
            while (j < n and _is_uncertain(words[j], threshold)
                   and not words[j].get("insert_after") and not words[j].get("insert_before")
                   and (j == i or speaker_of(words[j]) in (None, spk)) and (j == i or j not in loops)):
                j += 1
            run = words[i:j]
            if len(run) >= run_min:
                body, tail = _split_trailing(" ".join(w["text"] for w in run))
                blocks[-1].append(f"[?: {body}]{tail}")
                i = j
                continue
            blocks[-1].extend(render_word(w, threshold) for w in run)
            i = j
            continue
        blocks[-1].append(_with_inserts(word, render_word(word, threshold)))
        i += 1
    stamps = [""] * len(blocks)
    if stamp_every_s and words:
        long_form = (words[-1].get("end_ms") or 0) >= 3_600_000
        last = 0
        for k, start in enumerate(starts):
            if start is not None and blocks[k] and start - last >= stamp_every_s * 1000:
                stamps[k] = _clock(start, long_form) + " "
                last = start
    return [stamp + prefix + " ".join(b) for stamp, prefix, b in zip(stamps, prefixes, blocks) if b]


def stats(words: list[dict], threshold: float) -> dict:
    """Counts and an overall certainty level (share of marked words)."""
    total = len(words)
    repeated = {k for a, b in repetition_runs(words).items() for k in range(a, b)}
    uncertain = sum(1 for k, w in enumerate(words) if _is_uncertain(w, threshold) or k in repeated)
    alternatives = sum(1 for w in words if _has_alternative(w))
    inserts = sum(len(w.get("insert_after", [])) + len(w.get("insert_before", [])) for w in words)
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
        "reader_insertions": inserts,
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
#
# Two modes. Without a wake word, a sentence that starts with a marker word is a
# marker ("Idea: ..."), found in the rendered text. With a wake word, a marker
# needs the wake word right before the marker word ("Dictate idea: ..."), found in
# the words, so the trigger's confidence is known. Patterns may end in "*" to
# accept inflected forms ("question*" also matches "questions"). Punctuation or
# a pause may separate the wake word, the marker word and the body. A marker
# covers one sentence, or everything up to the end phrase (wake word + end word)
# when one follows before the next marker. A trigger heard with low confidence
# is not dropped: the marker is kept and flagged ``uncertain_trigger``.

MARKER_MAX_WORDS = 300
MARKER_MAX_CHARS = 2000
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")


@dataclass
class MarkerSpec:
    kinds: dict[str, str] = field(default_factory=dict)  # spoken pattern (lower case) -> label
    wake: list[str] = field(default_factory=list)  # wake word patterns
    end: list[str] = field(default_factory=list)  # end word patterns (after a wake word)


def _pattern_regex(patterns) -> str:
    return "|".join(sorted((re.escape(p.rstrip("*")) + (r"\w*" if p.endswith("*") else "")
                            for p in patterns), key=len, reverse=True))


def _match(word: str, patterns) -> str | None:
    """The pattern a single word matches (case-insensitive, punctuation ignored), else None."""
    token = _norm_word(word)
    for p in patterns:
        stem = p.rstrip("*").casefold()
        if token == stem or (p.endswith("*") and token.startswith(stem)):
            return p
    return None


def _shaky(word: dict, threshold: float) -> bool:
    return _is_uncertain(word, threshold) or _has_alternative(word)


def _markers_from_words(words: list[dict], spec: MarkerSpec, threshold: float) -> list[dict]:
    triggers = []  # (wake index, kind pattern or None for an end phrase, index of the kind/end word)
    for i, w in enumerate(words[:-1]):
        if not _match(w.get("text", ""), spec.wake):
            continue
        nxt = words[i + 1].get("text", "")
        kind = _match(nxt, spec.kinds)
        if kind:
            triggers.append((i, kind, i + 1))
        elif spec.end and _match(nxt, spec.end):
            triggers.append((i, None, i + 1))
    found = []
    for n, (wake_i, kind, kind_i) in enumerate(triggers):
        if kind is None:
            continue
        start = kind_i + 1
        nxt = triggers[n + 1] if n + 1 < len(triggers) else None
        if nxt and nxt[1] is None:  # an end phrase follows before the next marker
            stop, ended = nxt[0], True
        else:
            stop, ended = len(words), False
            # one sentence; if the marker word itself ends a sentence, the next one
            for k in range(start, min(stop, start + MARKER_MAX_WORDS)):
                if _SENTENCE_END.search(words[k].get("text", "")):
                    stop = k + 1
                    break
        stop = min(stop, start + MARKER_MAX_WORDS, nxt[0] if nxt else len(words))
        body_words = words[start:stop]
        body = " ".join(render_blocks(body_words, threshold, pause_ms=10 ** 9)).strip()
        body = body.lstrip(":,.;-–— ").rstrip()
        if not ended:
            body = body.rstrip(".!?… ")
        if len(body) < 3:
            continue
        marker = {"label": spec.kinds[kind], "text": body[:MARKER_MAX_CHARS],
                  "start_ms": words[wake_i].get("start_ms")}
        if ended:
            marker["ended_by_phrase"] = True
        if _shaky(words[wake_i], threshold) or _shaky(words[kind_i], threshold):
            marker["uncertain_trigger"] = True
        found.append(marker)
    return found


def extract_markers(text: str, markers, words: list[dict] | None = None, threshold: float = 0.7) -> list[dict]:
    """Find spoken markers. ``markers`` is a :class:`MarkerSpec` or a plain
    ``{spoken word: label}`` dict (no wake word)."""
    spec = markers if isinstance(markers, MarkerSpec) else MarkerSpec(kinds=dict(markers or {}))
    if not spec.kinds:
        return []
    if spec.wake:
        return _markers_from_words(words or [], spec, threshold)
    if not text:
        return []
    kinds = _pattern_regex(spec.kinds)
    pattern = re.compile(
        rf"(?:^|(?<=[.!?…\n]))\s*(?P<kw>{kinds})\b[\s:,.\-–]*"
        rf"(?P<body>(?:\[[^\]]*\]|\{{[^}}]*\}}|\(\?\)|[^.!?…\[{{\n])+[.!?…]?)",
        re.IGNORECASE,
    )
    found = []
    for match in pattern.finditer(text):
        body = match.group("body").strip().rstrip(".!?… ")
        kind = _match(match.group("kw"), spec.kinds)
        if len(body) >= 3 and kind:
            found.append({"label": spec.kinds[kind], "text": body[:400]})
    return found
