"""Short snippets of other voices are set aside, not written into the transcript (optional).

A dictation picks up voices that are not part of it: an assistant's speech from the phone's
speaker, a radio, someone passing. The engine gives each a speaker of its own, and a few seconds of
it land in the middle of the owner's sentence. With ``VOICEMEMOS_QUARANTINE_SHORT=1`` such a voice
is taken out of the transcript's words and listed apart.

The rule is about a voice, not about a turn: a speaker whose speech in the whole recording adds up
to less than ``VOICEMEMOS_QUARANTINE_MAX_S`` (5 s). A person the owner talks with says "yes" in
under a second many times and far more than five seconds in all, and stays. It applies only when
the voiceprint has found the owner among the speakers (a firm or a possible match): without that
nobody can say which voice is the foreign one, and nothing is set aside. The owner's own speaker is
never set aside, however little they say.

Nothing is dropped silently. The transcript says in one line how many snippets were set aside, the
JSON lists each under ``quarantine`` (speaker label, start and end, number of words), and the
engine's raw tokens (``source_tokens``) still hold the words themselves. The list carries no text
of its own, so a range taken out of the recording later (``redact``) leaves no copy behind here.
"""

from __future__ import annotations

DEFAULT_MAX_S = 5.0
GAP_MS = 2000  # words of one set-aside voice closer than this are one snippet


def _owner(speakers: dict) -> str | None:
    return next((key for key, value in speakers.items() if value.get("basis") == "voiceprint"), None)


def split(words: list[dict], speakers: dict, max_s: float = DEFAULT_MAX_S) -> tuple[list[dict], dict, dict | None]:
    """(words kept, speakers kept, quarantine record or None).

    ``speakers`` is the labelled speaker table after the voiceprint was applied. Speech time is the
    sum of the speaker's word spans; a word without times counts as nothing, so a speaker with no
    timed word at all is left alone (nothing is known about how long they spoke)."""
    owner = _owner(speakers or {})
    if owner is None or max_s <= 0:
        return words, speakers, None
    spoken: dict[str, int] = {}
    timed: dict[str, int] = {}
    for word in words:
        key = word.get("speaker")
        if key is None or word.get("start_ms") is None or word.get("end_ms") is None:
            continue
        spoken[key] = spoken.get(key, 0) + max(0, word["end_ms"] - word["start_ms"])
        timed[key] = timed.get(key, 0) + 1
    aside = {key for key in speakers if key != owner and timed.get(key) and spoken[key] < max_s * 1000
             and all(w.get("start_ms") is not None for w in words if w.get("speaker") == key)}
    if not aside:
        return words, speakers, None
    snippets: list[dict] = []
    for word in words:
        key = word.get("speaker")
        if key not in aside:
            continue
        last = snippets[-1] if snippets else None
        if last and last["_key"] == key and word["start_ms"] - last["end_ms"] <= GAP_MS:
            last.update(end_ms=word["end_ms"], words=last["words"] + 1)
        else:
            snippets.append({"_key": key, "speaker": speakers[key]["label"], "start_ms": word["start_ms"],
                             "end_ms": word["end_ms"], "words": 1})
    kept = [w for w in words if w.get("speaker") not in aside]
    total = sum(1 for w in kept if w.get("speaker") in speakers) or 1
    left = {}
    for key, value in speakers.items():
        if key in aside:
            continue
        count = sum(1 for w in kept if w.get("speaker") == key)
        left[key] = {**value, "words": count, "share": round(count / total, 3)}
    record = {"max_s": max_s, "voices": len(aside),
              "snippets": [{k: v for k, v in s.items() if k != "_key"} for s in snippets]}
    return kept, left, record


def line(record: dict | None) -> str | None:
    """The transcript's one line about what was set aside, or None."""
    if not record or not record.get("snippets"):
        return None
    n = len(record["snippets"])
    return (f"{n} short snippet{'s' if n != 1 else ''} of {record['voices']} other voice"
            f"{'s' if record['voices'] != 1 else ''} (under {record['max_s']:g} s of speech in the whole recording) "
            f"{'were' if n != 1 else 'was'} set aside and {'are' if n != 1 else 'is'} not in the text below; "
            "the JSON lists the times under `quarantine`.")
