"""An own speaker split on top of the engine's (optional, fully local).

The speech engine's speakers are often not people. Seen with Soniox v5 and v6 on meetings and
on a recorder's ambient recordings: two people under one number, one person under two numbers,
and a minute of one person's talk under another's number. The voice model already used for the
owner's voiceprint (``voiceprint.py``) tells voices apart better than that, so the engine's
speakers are taken as a first guess and looked at again:

1. Each speaker's talk is cut into units of a few seconds (at pauses and at other speakers'
   words), and every unit long enough gets an embedding.
2. Divide. A speaker's units are divided in two (spherical 2-means, from several starting
   points), and the division is kept when the two sides' average voices are less alike than
   one person's own talk is. Each side is looked at again the same way.
3. Join. Sides whose average voices are as alike as one person's are joined, whichever of the
   engine's speakers they came from.
4. Units too short for an embedding go with the nearest embedded unit of the same engine
   speaker in time.

Both decisions compare averages over many units, never single units: single units scatter
widely (a unit against its own speaker's average: 0.4 to 0.9), averages do not. On ten
recordings (26.11.2026: three meetings on Zoom, seven ambient recordings, 18 to 137 minutes)
two sides of at least 40 s each were mostly either at least 0.84 alike or at most 0.67; a few pairs
fell between (0.67 to 0.80), and there a decision can be wrong either way. That gap is where ``SIMILARITY`` sits. Averages over less speech are noisier, so a
small side is divided off only when it is very different (``DIVIDE_SMALL``) and joined only
when it is very alike (``JOIN_SMALL``); what falls between stays where the engine put it.

A recording where the engine was right comes out unchanged. The engine's own number stays in
the data: a word whose speaker changed carries ``speaker_engine``, and
``data["speaker_split"]`` says what was divided and joined.

Not solved by this: a person with only a few seconds of talk, and short interjections inside
someone else's turn (the engine does not mark them, and they are too short to embed). In a
recording transcribed in parts every part is done on its own, as the engine's numbers are.
"""

from __future__ import annotations

UNIT_MAX_S = 6.0          # a unit is cut when it would grow past this
UNIT_MIN_S = 1.5          # shorter units get no embedding of their own
UNIT_GAP_MS = 400         # a pause this long ends a unit
TRIM_S = 0.1              # cut from both ends of a unit before embedding
SIMILARITY = 0.72         # average voices less alike than this are two people, more alike are one
BIG_SIDE_S = 40.0         # ... when both have at least this much speech
DIVIDE_SMALL = 0.55       # a smaller side is divided off only below this
JOIN_SMALL = 0.80         # ... and joined only from this on
MIN_SIDE_S = 12.0         # speech needed on each side of a division, in embedded units
MIN_SIDE_UNITS = 4
MAX_DEPTH = 3             # one engine speaker is divided into at most 2**3
METHOD = "divide-join@1"


def different(similarity: float, seconds_a: float, seconds_b: float) -> bool:
    """Two sides of one engine speaker are two people."""
    return similarity < (SIMILARITY if min(seconds_a, seconds_b) >= BIG_SIDE_S else DIVIDE_SMALL)


def same(similarity: float, seconds_a: float, seconds_b: float) -> bool:
    """Two groups of units are one person."""
    return similarity >= (SIMILARITY if min(seconds_a, seconds_b) >= BIG_SIDE_S else JOIN_SMALL)


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("numpy is required (pip install 'voicememos-mcp[voiceprint]')") from exc
    return np


def units(words: list[dict]) -> list[dict]:
    """Stretches of one engine speaker's talk: ``{"first", "last" (exclusive), "speaker",
    "start_s", "end_s"}``. Words without times or without a speaker belong to no unit."""
    out: list[dict] = []
    current = None
    for k, word in enumerate(words):
        start, end, spk = word.get("start_ms"), word.get("end_ms"), word.get("speaker")
        if start is None or end is None or spk in (None, ""):
            current = None
            continue
        spk = str(spk)
        if (current is not None and current["speaker"] == spk and current["last"] == k
                and start - current["end_s"] * 1000 <= UNIT_GAP_MS
                and end / 1000 - current["start_s"] <= UNIT_MAX_S):
            current.update(last=k + 1, end_s=end / 1000)
        else:
            current = {"first": k, "last": k + 1, "speaker": spk, "start_s": start / 1000, "end_s": end / 1000}
            out.append(current)
    return out


def _unit(v):
    np = _np()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def embed_units(audio, found: list[dict], embed) -> None:
    """Add ``vector`` (unit length) to every unit long enough; the others stay without one."""
    from .voiceprint import SAMPLE_RATE, as_reader

    read = as_reader(audio)
    for unit in found:
        length = unit["end_s"] - unit["start_s"]
        if length < UNIT_MIN_S:
            continue
        samples = read(unit["start_s"] + TRIM_S, length - 2 * TRIM_S)
        if len(samples) >= int((UNIT_MIN_S - 2 * TRIM_S) * SAMPLE_RATE) - 1:
            unit["vector"] = _unit(embed(samples))


def _two_means(vectors, weights, seed_a: int, seed_b: int):
    np = _np()
    centres = np.stack([vectors[seed_a], vectors[seed_b]])
    labels = None
    for _ in range(30):
        new = (vectors @ centres.T).argmax(axis=1)
        if labels is not None and (new == labels).all():
            break
        labels = new
        for side in (0, 1):
            mask = labels == side
            if not mask.any():
                return None
            centres[side] = _unit((vectors[mask] * weights[mask, None]).sum(axis=0))
    return labels


def bisect(vectors, weights) -> dict | None:
    """The best division of the units in two: ``{"labels", "similarity", "seconds": [a, b]}``,
    the one whose sides' average voices are least alike among those where both sides have
    enough speech. None when no division leaves enough on both sides."""
    np = _np()
    n = len(vectors)
    if n < 2 * MIN_SIDE_UNITS:
        return None
    mean = _unit((vectors * weights[:, None]).sum(axis=0))
    starts = []
    far = int((vectors @ mean).argmin())
    starts.append((far, int((vectors @ vectors[far]).argmin())))
    centred = vectors - mean
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    for axis in axes[:3]:  # along the directions in which the units differ most
        projection = centred @ axis
        starts.append((int(projection.argmax()), int(projection.argmin())))
    order = np.argsort(-weights)  # the two longest units that are least alike
    longest = order[: min(n, 12)]
    sims = vectors[longest] @ vectors[longest].T
    a, b = np.unravel_index(int(sims.argmin()), sims.shape)
    starts.append((int(longest[a]), int(longest[b])))
    best = None
    for seed_a, seed_b in starts:
        if seed_a == seed_b:
            continue
        labels = _two_means(vectors, weights, seed_a, seed_b)
        if labels is None:
            continue
        seconds = [float(weights[labels == side].sum()) for side in (0, 1)]
        counts = [int((labels == side).sum()) for side in (0, 1)]
        if min(seconds) < MIN_SIDE_S or min(counts) < MIN_SIDE_UNITS:
            continue
        centres = [_unit((vectors[labels == side] * weights[labels == side, None]).sum(axis=0)) for side in (0, 1)]
        similarity = float(centres[0] @ centres[1])
        if best is None or similarity < best["similarity"]:
            best = {"labels": labels, "similarity": round(similarity, 4), "seconds": [round(s, 1) for s in seconds]}
    return best


def _divide(vectors, weights, indexes, depth: int, tried: list[dict]) -> list:
    """Groups of unit indexes: the given units divided as long as the sides are different people."""
    if depth >= MAX_DEPTH:
        return [indexes]
    found = bisect(vectors[indexes], weights[indexes])
    divided = found is not None and different(found["similarity"], *found["seconds"])
    if found is not None:
        tried.append({"similarity": found["similarity"], "seconds": found["seconds"], "divided": divided})
    if not divided:
        return [indexes]
    groups = []
    for side in (0, 1):
        groups += _divide(vectors, weights, indexes[found["labels"] == side], depth + 1, tried)
    return groups


def _prefix(key: str) -> tuple[str, int | None]:
    """A speaker key is a number, or ``part:number`` in a recording transcribed in parts."""
    head, _, tail = key.rpartition(":")
    return (head + ":" if head else "", int(tail) if tail.isdigit() else None)


def _regroup_part(mine: list[dict], prefix: str) -> dict | None:
    """Divide and join the units of one part (all of a recording sent in one file). Sets
    ``unit["new"]`` on every unit and returns what was done, or None when nothing changed."""
    np = _np()
    embedded = [u for u in mine if "vector" in u]
    by_speaker: dict[str, list[int]] = {}
    for i, unit in enumerate(embedded):
        by_speaker.setdefault(unit["speaker"], []).append(i)
    vectors = np.stack([u["vector"] for u in embedded]) if embedded else None
    weights = np.array([u["end_s"] - u["start_s"] for u in embedded], dtype=np.float32)
    groups: list[dict] = []  # {"units": [index], "sum": weighted vector sum, "seconds", "from": {engine key: seconds}}
    divisions = []
    for key in sorted(by_speaker, key=lambda k: (_prefix(k)[1] or 0, k)):
        indexes = np.array(by_speaker[key])
        tried: list[dict] = []
        parts = _divide(vectors, weights, indexes, 0, tried) if len(indexes) >= 2 * MIN_SIDE_UNITS else [indexes]
        if len(parts) > 1:
            divisions.append({"engine_speaker": key, "sides": len(parts),
                              "similarity": min(t["similarity"] for t in tried if t["divided"])})
        for part in parts:
            seconds = float(weights[part].sum())
            groups.append({"units": [int(i) for i in part], "seconds": seconds, "from": {key: seconds},
                           "sum": (vectors[part] * weights[part, None]).sum(axis=0)})
    joins = []
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a):
                similarity = float(_unit(groups[a]["sum"]) @ _unit(groups[b]["sum"]))
                if same(similarity, groups[a]["seconds"], groups[b]["seconds"]) and (best is None or similarity > best[0]):
                    best = (similarity, a, b)
        if best is None:
            break
        similarity, a, b = best
        one, other = groups[b], groups.pop(a)
        if set(one["from"]) != set(other["from"]) or len(one["from"]) > 1:
            joins.append(round(similarity, 4))
        one["units"] += other["units"]
        one["sum"] = one["sum"] + other["sum"]
        one["seconds"] += other["seconds"]
        for key, seconds in other["from"].items():
            one["from"][key] = one["from"].get(key, 0.0) + seconds
    # Numbers: as much talk as possible keeps the engine's number (the biggest share first); a
    # group left without one gets a new number.
    groups.sort(key=lambda g: -g["seconds"])
    used: set[str] = set()
    shares = sorted(((seconds, n, key) for n, g in enumerate(groups) for key, seconds in g["from"].items()),
                    key=lambda share: (-share[0], share[1], share[2]))
    for _, n, key in shares:
        if "key" not in groups[n] and key not in used:
            groups[n]["key"] = key
            used.add(key)
    highest = max([_prefix(u["speaker"])[1] or 0 for u in mine], default=0)
    for group in groups:
        if "key" not in group:
            highest += 1
            group["key"] = f"{prefix}{highest}"
        for i in group["units"]:
            embedded[i]["new"] = group["key"]
    anchors: dict[str, list[dict]] = {}
    for unit in embedded:
        anchors.setdefault(unit["speaker"], []).append(unit)
    for unit in mine:  # too short to embed: with the nearest embedded unit of the same engine speaker
        if "new" not in unit:
            near = anchors.get(unit["speaker"])
            if not near:
                unit["new"] = unit["speaker"]
                continue
            middle = (unit["start_s"] + unit["end_s"]) / 2
            unit["new"] = min(near, key=lambda u: min(abs(u["start_s"] - middle), abs(u["end_s"] - middle)))["new"]
    if all(unit["new"] == unit["speaker"] for unit in mine):
        return None
    closest = max((float(_unit(groups[a]["sum"]) @ _unit(groups[b]["sum"]))
                   for a in range(len(groups)) for b in range(a)), default=None)
    return {"divided": divisions, "joined": joins,
            "speakers": [{"speaker": g["key"], "seconds": round(g["seconds"], 1),
                          "from": {k: round(v, 1) for k, v in sorted(g["from"].items())}} for g in groups],
            "closest_similarity": round(closest, 4) if closest is not None else None}


def regroup(words: list[dict], found: list[dict]) -> tuple[list[dict], dict]:
    """``split`` for units that already have their embeddings."""
    words = [dict(w) for w in words]
    parts: dict[str, list[dict]] = {}
    for unit in found:
        parts.setdefault(_prefix(unit["speaker"])[0], []).append(unit)
    summary = {"method": METHOD, "engine_speakers": len({u["speaker"] for u in found}), "changed_words": 0}
    done = []
    for prefix in sorted(parts):
        result = _regroup_part(parts[prefix], prefix)
        if result is None:
            continue
        done.append({**({"part": int(prefix[:-1])} if prefix[:-1].isdigit() else {}), **result})
        for unit in parts[prefix]:
            if unit["new"] != unit["speaker"]:
                for word in words[unit["first"]:unit["last"]]:
                    word["speaker_engine"] = word["speaker"]
                    word["speaker"] = unit["new"]
                    summary["changed_words"] += 1
    if done:
        summary["parts" if any("part" in d for d in done) else "result"] = done if any("part" in d for d in done) else done[0]
    summary["speakers"] = len({str(w["speaker"]) for w in words if w.get("speaker") not in (None, "")})
    return words, summary


def restore(words: list[dict]) -> list[dict]:
    """The words with the engine's own speakers again."""
    out = []
    for word in words:
        word = dict(word)
        if "speaker_engine" in word:
            word["speaker"] = word.pop("speaker_engine")
        out.append(word)
    return out


def split(audio, words: list[dict], embed) -> tuple[list[dict], dict]:
    """Return the words with the speakers divided and joined by voice, and the summary for
    ``data["speaker_split"]``. A word whose speaker changed carries ``speaker_engine``. Running
    it again starts from the engine's own speakers."""
    words = restore(words)
    found = units(words)
    embed_units(audio, found, embed)
    return regroup(words, found)
