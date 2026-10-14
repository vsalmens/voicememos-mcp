"""Compare the primary engine with the second reader, separately per language.

Agreement says how often two engines heard the same thing, not which one is
right. The report therefore also lists a random sample of disagreements with
time offsets, so a person can listen and judge (the only real ground truth).

It also checks calibration: if the primary engine's confidence is meaningful,
disagreements should cluster on words it already marked as uncertain.
"""

from __future__ import annotations

import datetime as dt
import random
import statistics
from collections import defaultdict

from . import align
from .store import Store, now_iso, write_json_atomic

LANGUAGE_LABELS = {"fi": "Finnish", "en": "English", "mixed": "Mixed", "unknown": "Unknown"}


def _fmt_time(ms) -> str:
    if ms is None:
        return "?"
    seconds = int(ms // 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _disagreements(data: dict, context: int = 5) -> list[dict]:
    words = data.get("words", [])
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        span = None
        if w.get("alts"):
            span = (i, i + 1, w["alts"][0])
        elif w.get("span"):
            span = (i, i + max(w["span"].get("n", 1), 1), w["span"]["alts"][0])
        for key in ("insert_before", "insert_after"):
            for alt in w.get(key, []):
                pos = i if key == "insert_before" else i + 1
                out.append({"start_ms": w.get("start_ms"), "primary": "∅", "reader": alt,
                            "primary_conf": None,
                            "context": " ".join(x["text"] for x in words[max(0, pos - context):pos]) + " ▮ "
                            + " ".join(x["text"] for x in words[pos:pos + context])})
        if span:
            a, b, alt = span
            confs = [x.get("conf") for x in words[a:b] if x.get("conf") is not None]
            out.append({
                "start_ms": words[a].get("start_ms"),
                "primary": " ".join(x["text"] for x in words[a:b]),
                "reader": alt or "∅",
                "primary_conf": round(min(confs), 2) if confs else None,
                "context": " ".join(x["text"] for x in words[max(0, a - context):a]) + " ▮ "
                + " ".join(x["text"] for x in words[b:b + context]),
            })
            i = b
            continue
        i += 1
    return out


def build_report(store: Store, since: str | None = None, until: str | None = None, samples: int = 15,
                 seed: int = 1) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for memo in store.read_library().get("memos", []):
        day = (memo.get("recorded") or "")[:10]
        if (since and day < since) or (until and day > until):
            continue
        transcript = memo.get("transcript") or {}
        if not transcript.get("json"):
            continue
        data = store.load_transcript(transcript["json"])
        if not data:
            continue
        second = next((r for r in data.get("readers", []) if r.get("role") == "second" and not r.get("error")), None)
        if not second:
            continue
        if second.get("text"):  # re-align with the current comparison rules (e.g. spoken-form folding)
            words, metrics = align.align(data.get("words", []), second["text"],
                                         (data.get("stats") or {}).get("threshold") or 0.7)
            data = {**data, "words": words}
            second = {**second, "agreement": metrics}
        groups[data.get("language") or "unknown"].append({"memo": memo, "data": data, "second": second})

    rng = random.Random(seed)
    languages = {}
    for lang, items in sorted(groups.items()):
        metrics = [i["second"]["agreement"] for i in items]
        primary_words = sum(m["primary_words"] for m in metrics)
        spans = sum(m["disagreement_spans"] for m in metrics)
        spans_low = sum(m["spans_with_low_primary_confidence"] for m in metrics)
        low_words = sum(m["low_confidence_words"] for m in metrics)
        low_confirmed = sum(m["low_confidence_words_confirmed_by_reader"] for m in metrics)
        tokens = sum((i["second"].get("usage") or {}).get("totalTokenCount", 0) for i in items)
        pool = []
        for item in items:
            for d in _disagreements(item["data"]):
                pool.append({**d, "memo_id": item["memo"].get("short_id"), "title": item["memo"].get("title"),
                             "recorded": (item["memo"].get("recorded") or "")[:16]})
        languages[lang] = {
            "memos": len(items),
            "primary_words": primary_words,
            "reader_words": sum(m["reader_words"] for m in metrics),
            "agreement_mean": round(statistics.mean(m["agreement"] for m in metrics), 3),
            "agreement_median": round(statistics.median(m["agreement"] for m in metrics), 3),
            "word_difference_rate": round(sum(m["substituted"] + m["deleted"] + m["inserted"] for m in metrics)
                                          / primary_words, 3) if primary_words else None,
            "disagreements_per_100_words": round(100 * spans / primary_words, 1) if primary_words else None,
            "disagreements_on_low_confidence_words": round(spans_low / spans, 3) if spans else None,
            "low_confidence_words_confirmed_by_reader": round(low_confirmed / low_words, 3) if low_words else None,
            "reader_unsure_marks_per_100_words": round(100 * sum(m["reader_unsure_marks"] for m in metrics)
                                                       / primary_words, 1) if primary_words else None,
            "reader_tokens": tokens,
            "models": sorted({i["second"].get("model") or "?" for i in items}),
            # listening order: one recording at a time, front to back
            "samples": sorted(rng.sample(pool, min(samples, len(pool))),
                              key=lambda d: (d["recorded"], d["memo_id"] or "", d["start_ms"] or 0)),
        }
    return {"created": now_iso(), "since": since, "until": until, "languages": languages}


def render_report(report: dict) -> str:
    lines = ["# Second reader comparison", "",
             f"Created {report['created'][:16].replace('T', ' ')}."
             + (f" Memos from {report['since']}" if report.get("since") else "")
             + (f" to {report['until']}" if report.get("until") else ""), "",
             "Agreement shows how often the engines heard the same words, not who is right. "
             "Samples are grouped by recording and in time order, so each recording is opened once. "
             "Listen (time = offset in the memo) and tick the better reading.", ""]
    if not report["languages"]:
        return "\n".join(lines + ["No memos with a second reader yet."]) + "\n"
    rows = [("Memos", "memos"), ("Primary words", "primary_words"), ("Reader words", "reader_words"),
            ("Word agreement (mean)", "agreement_mean"), ("Word agreement (median)", "agreement_median"),
            ("Word difference rate", "word_difference_rate"),
            ("Disagreements per 100 words", "disagreements_per_100_words"),
            ("Disagreements on words the primary engine marked uncertain", "disagreements_on_low_confidence_words"),
            ("Uncertain primary words the reader confirmed", "low_confidence_words_confirmed_by_reader"),
            ("Reader's own (?) marks per 100 words", "reader_unsure_marks_per_100_words"),
            ("Reader tokens used", "reader_tokens")]
    langs = list(report["languages"])
    lines.append("| | " + " | ".join(LANGUAGE_LABELS.get(lang, lang) for lang in langs) + " |")
    lines.append("|---|" + "---|" * len(langs))
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(str(report["languages"][lang].get(key)) for lang in langs) + " |")
    lines.append("")
    for lang in langs:
        group = report["languages"][lang]
        lines += [f"## Samples: {LANGUAGE_LABELS.get(lang, lang)} ({', '.join(group['models'])})", ""]
        current = None
        for s in group["samples"]:
            if s["memo_id"] != current:
                current = s["memo_id"]
                lines += ["", f"### {s.get('title') or '(untitled)'}, {s['recorded'].replace('T', ' ')} `{current}`", ""]
            conf = f", primary confidence {s['primary_conf']}" if s.get("primary_conf") is not None else ""
            lines += [f"- [ ] at {_fmt_time(s['start_ms'])}{conf}",
                      f"  - A (primary): {s['primary']}",
                      f"  - B (reader): {s['reader']}",
                      f"  - context: {s['context']}",
                      "  - better: A / B / neither"]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(store: Store, report: dict) -> dict:
    folder = store.output_dir / "_reports"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    md = folder / f"second-reader_{stamp}.md"
    md.write_text(render_report(report), encoding="utf-8")
    write_json_atomic(md.with_suffix(".json"), report)
    return {"markdown": str(md), "json": str(md.with_suffix(".json"))}
