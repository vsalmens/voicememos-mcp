"""Other formats made from the word data: WebVTT (subtitles) and TEI (spoken text).

Both are rendered on demand from the transcript JSON and never stored. Words set aside
(``quarantine``) are left out, as in the Markdown; a range taken out of the recording is a gap.

* **WebVTT**: one cue per stretch of at most ``CUE_MAX_S`` seconds, cut at a change of speaker,
  at a pause and after a sentence. The speaker is a voice tag (``<v Name>``). The uncertainty
  marks stay in the text as the transcript writes them.
* **TEI** (P5, chapter 8 "Transcriptions of Speech"): one ``<u>`` per turn with ``who``,
  ``start`` and ``end`` pointing into a ``<timeline>``, ``<w>`` per word, ``<pause>`` for gaps,
  ``<unclear>`` for a word under the confidence threshold (``cert`` = the engine's confidence),
  ``<choice>`` with one ``<seg>`` per reading where the readers disagree (``resp`` says whose;
  ``ana="#picked"`` on the one a third listening sided with), ``<gap reason="redacted">``.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from . import notation

CUE_MAX_S = 7.0
CUE_PAUSE_MS = 700
TURN_PAUSE_MS = 2500
PAUSE_MS = 500


def _vtt_time(ms: int) -> str:
    ms = max(int(ms), 0)
    return f"{ms // 3_600_000:02d}:{ms % 3_600_000 // 60_000:02d}:{ms % 60_000 // 1000:02d}.{ms % 1000:03d}"


def _items(words: list[dict], threshold: float) -> list[dict]:
    """Rendered pieces with their time and speaker: a word, or the words of one disagreement."""
    words = notation.shown(words)
    loops = notation.repetition_runs(words)
    out = []
    i, n = 0, len(words)
    while i < n:
        word = words[i]
        if i in loops:
            group = words[i: loops[i]]
            body, tail = notation._split_trailing(group[-1]["text"])
            text = f"[?: {body} ×{len(group)}]{tail}"
        elif word.get("span"):
            span = word["span"]
            group = words[i: i + max(span.get("n", 1), 1)]
            body, tail = notation._split_trailing(" ".join(w["text"] for w in group)) if span.get("n", 1) else ("", "")
            text = notation._with_inserts(
                {"insert_before": word.get("insert_before", []), "insert_after": group[-1].get("insert_after", []),
                 "adj_before": word.get("adj_before"), "adj_after": group[-1].get("adj_after")},
                notation._choice([body] + list(span.get("alts", [])), tail, word.get("adj")))
        else:
            group = [word]
            text = notation._with_inserts(word, notation.render_word(word, threshold))
        starts = [w["start_ms"] for w in group if w.get("start_ms") is not None]
        ends = [w["end_ms"] for w in group if w.get("end_ms") is not None]
        out.append({"text": text, "words": group, "start_ms": starts[0] if starts else None,
                    "end_ms": ends[-1] if ends else None, "speaker": word.get("speaker")})
        i += len(group)
    return out


def _label(speakers: dict, speaker) -> str | None:
    entry = (speakers or {}).get(str(speaker)) if speaker not in (None, "") else None
    return entry["label"] if entry else None


def webvtt(data: dict, threshold: float) -> str:
    speakers = data.get("speakers") or {}
    cues: list[dict] = []
    for item in _items(data.get("words", []), threshold):
        if item["start_ms"] is None or item["end_ms"] is None:
            if cues:  # no time of its own: with the words before it
                cues[-1]["text"].append(item["text"])
            continue
        last = cues[-1] if cues else None
        new = (last is None or last["speaker"] != item["speaker"]
               or item["start_ms"] - last["end_ms"] > CUE_PAUSE_MS
               or item["end_ms"] - last["start_ms"] > CUE_MAX_S * 1000
               or (last["text"] and notation._SENTENCE_END.search(last["text"][-1])
                   and item["end_ms"] - last["start_ms"] > CUE_MAX_S * 500))
        if new:
            cues.append({"start_ms": item["start_ms"], "end_ms": item["end_ms"], "speaker": item["speaker"],
                         "text": [item["text"]]})
        else:
            last["text"].append(item["text"])
            last["end_ms"] = max(last["end_ms"], item["end_ms"])
    for redaction in data.get("redactions") or []:
        cues.append({"start_ms": redaction["start_ms"], "end_ms": redaction["end_ms"], "speaker": None,
                     "text": [notation.redaction_line(redaction)]})
    cues.sort(key=lambda c: c["start_ms"])
    memo = data.get("memo") or {}
    lines = ["WEBVTT", "", f"NOTE {memo.get('title') or 'Voice memo'}, recorded {memo.get('recorded') or ''}", ""]
    for number, cue in enumerate(cues, start=1):
        end = max(cue["end_ms"], cue["start_ms"] + 1)
        text = escape(" ".join(cue["text"]))
        label = _label(speakers, cue["speaker"])
        if label:
            text = f"<v {escape(label)}>{text}"
        lines += [str(number), f"{_vtt_time(cue['start_ms'])} --> {_vtt_time(end)}", text, ""]
    return "\n".join(lines)


def _seconds(ms: int) -> str:
    return f"{ms / 1000:.3f}".rstrip("0").rstrip(".") + "s"


def _tei_word(word: dict, threshold: float) -> str:
    text = word.get("text", "")
    if not text:
        return "<gap reason=\"inaudible\"/>"
    body = f"<w>{escape(text)}</w>"
    conf = word.get("conf")
    if conf is not None and conf < threshold:
        body = f"<unclear cert={quoteattr(f'{conf:.2f}')}>{body}</unclear>"
    return body


def _tei_choice(primary: str, second: str, adj: dict | None) -> str:
    pick = (adj or {}).get("pick")

    def seg(resp: str, text: str, picked: bool) -> str:
        ana = ' ana="#picked"' if picked else ""
        return f'<seg type="reading" resp="#{resp}"{ana}>{escape(text)}</seg>'

    parts = [seg("primary", primary, pick == 0), seg("second", second, pick == 1)]
    if pick == "other" and (adj or {}).get("heard"):
        parts.append(seg("adjudicator", str(adj["heard"]), True))
    return "<choice>" + "".join(parts) + "</choice>"


def tei(data: dict, threshold: float) -> str:
    speakers = data.get("speakers") or {}
    words = notation.shown(data.get("words", []))
    memo = data.get("memo") or {}
    whens: dict[int, str] = {}

    def when(ms: int) -> str:
        return whens.setdefault(int(ms), f"T{len(whens) + 1}")

    def who(speaker) -> str:
        return "spk" + "".join(ch if ch.isalnum() else "_" for ch in str(speaker))

    turns: list[str] = []
    current: list[str] = []
    turn = {"speaker": None, "start": None, "end": None}
    pending = sorted((r for r in data.get("redactions") or [] if isinstance(r, dict)), key=lambda r: r["start_ms"])

    def flush() -> None:
        nonlocal current
        if current:
            attrs = ""
            if turn["speaker"] not in (None, "") and str(turn["speaker"]) in speakers:
                attrs += f' who="#{who(turn["speaker"])}"'
            if turn["start"] is not None:
                attrs += f' start="#{when(turn["start"])}"'
            if turn["end"] is not None:
                attrs += f' end="#{when(turn["end"])}"'
            turns.append(f"   <u{attrs}>" + " ".join(current) + "</u>")
        current = []
        turn.update(speaker=None, start=None, end=None)

    previous_end = None
    i, n = 0, len(words)
    while i < n:
        word = words[i]
        start = word.get("start_ms")
        while pending and start is not None and start >= pending[0]["start_ms"]:
            flush()
            r = pending.pop(0)
            turns.append(f'   <gap reason="redacted" start="#{when(r["start_ms"])}" end="#{when(r["end_ms"])}"/>')
            previous_end = None
        gap = start - previous_end if start is not None and previous_end is not None else None
        if current and (word.get("speaker") != turn["speaker"] or (gap is not None and gap >= TURN_PAUSE_MS)):
            flush()
            if gap is not None and gap >= PAUSE_MS:
                turns.append(f'   <pause dur="PT{gap / 1000:.1f}S"/>')
        elif current and gap is not None and gap >= PAUSE_MS:
            current.append(f'<pause dur="PT{gap / 1000:.1f}S"/>')
        if not current:
            turn.update(speaker=word.get("speaker"), start=start)
        for alt in word.get("insert_before", []):
            current.append(_tei_choice("", alt, word.get("adj_before")))
        if word.get("span"):
            span = word["span"]
            group = words[i: i + max(span.get("n", 1), 1)]
            primary = " ".join(w.get("text", "") for w in group) if span.get("n", 1) else ""
            current.append(_tei_choice(primary, (span.get("alts") or [""])[0], word.get("adj")))
        elif word.get("alts"):
            group = [word]
            current.append(_tei_choice(word.get("text", ""), word["alts"][0], word.get("adj")))
        else:
            group = [word]
            current.append(_tei_word(word, threshold))
        for alt in group[-1].get("insert_after", []):
            current.append(_tei_choice("", alt, group[-1].get("adj_after")))
        ends = [w["end_ms"] for w in group if w.get("end_ms") is not None]
        if ends:
            previous_end = ends[-1]
            turn["end"] = ends[-1]
        i += max(len(group), 1)
    flush()
    for r in pending:
        turns.append(f'   <gap reason="redacted" start="#{when(r["start_ms"])}" end="#{when(r["end_ms"])}"/>')

    engine = data.get("engine") or {}
    readers = [r for r in data.get("readers") or [] if isinstance(r, dict) and not r.get("error")]
    resp = [f'    <respStmt xml:id="primary"><resp>primary transcription</resp><name>'
            f'{escape(str(engine.get("provider") or ""))} {escape(str(engine.get("model") or ""))}</name></respStmt>']
    for reader in readers:
        role = {"second": "second", "adjudicator": "adjudicator"}.get(reader.get("role"))
        if role:
            resp.append(f'    <respStmt xml:id="{role}"><resp>{"second reading" if role == "second" else "third listening"}'
                        f'</resp><name>{escape(str(reader.get("model") or ""))}</name></respStmt>')
    persons = [f'     <person xml:id="{who(key)}"><persName>{escape(str(value.get("label")))}</persName></person>'
               for key, value in speakers.items() if not value.get("quarantine")]
    timeline = [f'   <when xml:id="{name}" absolute="{_seconds(ms)}"/>' for ms, name in sorted(whens.items())]
    recorded = memo.get("recorded") or ""
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
        ' <teiHeader>',
        '  <fileDesc>',
        '   <titleStmt>',
        f'    <title>{escape(str(memo.get("title") or "Voice memo"))}</title>',
        *resp,
        '   </titleStmt>',
        '   <publicationStmt><p>Not published.</p></publicationStmt>',
        '   <sourceDesc>',
        f'    <recordingStmt><recording type="audio" dur="PT{int(memo.get("duration_s") or 0)}S">'
        f'<date when={quoteattr(recorded)}/></recording></recordingStmt>',
        '   </sourceDesc>',
        '  </fileDesc>',
        '  <encodingDesc>',
        f'   <p>Made from word-level data ({escape(str(data.get("schema") or ""))}). A word under confidence '
        f'{threshold} is in unclear with the engine\'s confidence as cert. Where two readers wrote different things, '
        'choice holds one seg per reading; ana="#picked" marks the one a third listening sided with.</p>',
        '   <classDecl><taxonomy><category xml:id="picked"><catDesc>the reading a third listening sided with'
        '</catDesc></category></taxonomy></classDecl>',
        '  </encodingDesc>',
    ]
    if persons:
        head += ['  <profileDesc><particDesc><listPerson>', *persons, '  </listPerson></particDesc></profileDesc>']
    head += [' </teiHeader>', ' <text>', '  <body>', '   <timeline unit="s">' if timeline else '']
    body = ([*timeline, '   </timeline>'] if timeline else []) + turns
    return "\n".join([line for line in head if line] + body + ['  </body>', ' </text>', '</TEI>', ''])
