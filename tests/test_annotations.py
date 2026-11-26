"""Named voices, quarantine, the third listening, the exports, and the passes run through the worker."""
import json

import pytest

np = pytest.importorskip("numpy")

from tests.conftest import FakeTranscriber  # noqa: E402
from tests.test_voiceprint import OTHER, OWNER, SR, fake_embed, tone, words_for  # noqa: E402
from voicememos_mcp import adjudicate, align, export, notation, voiceprint  # noqa: E402
from voicememos_mcp.store import render_markdown  # noqa: E402
from voicememos_mcp.tools import ToolError, Tools  # noqa: E402
from voicememos_mcp.worker import Worker  # noqa: E402

THIRD = 650


def enrolled():
    return voiceprint.enroll("Alex", "test-model", {f"m{i}": tone(OWNER, 15, seed=i) for i in range(3)}, fake_embed)


def with_voice(vp, name, freq, show=True, seed=30):
    words = words_for([("1", 0, 12)])
    vp.setdefault("others", []).append(voiceprint.other_voice(name, tone(freq, 12, seed=seed), words, "1", fake_embed,
                                                              show=show))
    return vp


# ----------------------------------------------------------------------------- named voices

def test_a_named_voice_labels_its_speaker_and_a_not_me_voice_never_does():
    vp = with_voice(with_voice(enrolled(), "Mira", OTHER), "Hidden", THIRD, show=False, seed=31)
    audio = np.concatenate([tone(OWNER, 8, seed=1), tone(OTHER, 8, seed=2), tone(THIRD, 8, seed=3)])
    words = words_for([("1", 0, 8), ("2", 8, 16), ("3", 16, 24)])
    updated, summary = voiceprint.identify(audio, words, notation.speaker_labels(words), vp, fake_embed)
    assert [updated[k]["label"] for k in "123"] == ["Alex", "Mira", "S3"]
    assert updated["2"]["basis"] == "voiceprint" and summary["named"]["2"]["firm"] and summary["speaker"] == "1"
    data = {"memo": {"id": "x", "title": "t", "recorded": "2026-11-26T10:00:00+02:00", "duration_s": 24},
            "words": words, "speakers": updated, "voiceprint": summary, "confidence_available": True}
    md = render_markdown(data, 0.7)
    assert "**Mira:**" in md and "**Alex:**" in md and "Hidden" not in md


def test_a_name_that_does_not_stand_out_gets_a_question_mark_and_one_name_goes_to_one_speaker():
    vp = with_voice(enrolled(), "Mira", OTHER)
    # Two speakers who both sound like the stored voice: the better one is named, with a doubt.
    audio = np.concatenate([tone(OTHER, 8, seed=2), tone(OTHER + 6, 8, seed=3)])
    words = words_for([("1", 0, 8), ("2", 8, 16)])
    updated, summary = voiceprint.identify(audio, words, notation.speaker_labels(words), vp, fake_embed)
    labels = sorted(v["label"] for v in updated.values())
    assert labels in (["Mira (?)", "S2"], ["Mira (?)", "S1"]) and summary["result"] == "no match"


def test_a_single_speaker_memo_says_whose_voice_it_sounds_like():
    vp = with_voice(enrolled(), "Mira", OTHER)
    _, summary = voiceprint.identify(tone(OTHER, 10, seed=5), words_for([("1", 0, 10)]), {}, vp, fake_embed)
    assert summary["result"] == "no match" and summary["named"]["name"] == "Mira"
    _, own = voiceprint.identify(tone(OWNER, 10, seed=6), words_for([("1", 0, 10)]), {}, vp, fake_embed)
    assert own["result"] == "match" and "named" not in own


def test_voice_names_are_checked(cfg, library):
    Worker(cfg, transcriber=FakeTranscriber()).run()
    tools = Tools(cfg)
    tools.store.voiceprint_path.write_text("{}")
    for bad in ("S2", "a:b", "x" * 41, ""):
        with pytest.raises(ToolError):
            tools.voiceprint("add_voice", memos=["aaaa1111"], name=bad)
    job = tools.voiceprint("add_voice", memos=["aaaa1111"], name="Mira")
    request = tools.transcribe_status(job["job_id"])["request"]
    assert request["action"] == "add_other" and request["show"] is True


# ----------------------------------------------------------------------------- quarantine

def test_a_few_seconds_of_an_unknown_voice_are_set_aside_only_beside_the_owner():
    words = words_for([("1", 0, 20), ("2", 20, 23), ("1", 23, 30), ("3", 30, 40)])
    speakers = notation.speaker_labels(words)
    speakers["1"].update(label="Alex", basis="voiceprint")
    out, summary = voiceprint.quarantine(words, speakers, 5.0)
    assert summary == {"max_s": 5.0, "speakers": 1, "words": 3, "seconds": 2.7}
    assert [w.get("quarantine", False) for w in out].count(True) == 3 and speakers["2"]["quarantine"]
    assert "quarantine" not in speakers["3"]  # ten seconds of talk is somebody, not noise
    text = "\n".join(notation.render_blocks(out, 0.7, speakers=speakers))
    assert "**S2:**" not in text and "**S3:**" in text and notation.stats(out, 0.7)["words"] == 37
    data = {"memo": {"id": "x", "title": "t", "recorded": "2026-11-26T10:00:00+02:00", "duration_s": 40},
            "words": out, "speakers": speakers, "quarantine": summary, "confidence_available": True}
    md = render_markdown(data, 0.7)
    assert "3 words of 1 unknown voice(s) that said under 5 s" in md and "S2 " not in md.split("## Transcript")[0]
    # Without a recognised owner nothing is unknown "beside" anybody; and off means off.
    plain = notation.speaker_labels(words)
    assert voiceprint.quarantine(words, plain, 5.0)[1] is None
    again, none = voiceprint.quarantine(out, speakers, 0)
    assert none is None and not any(w.get("quarantine") for w in again) and "quarantine" not in speakers["2"]


# ----------------------------------------------------------------------------- third listening

def disputed():
    primary = [{"text": w, "start_ms": k * 500, "end_ms": k * 500 + 400, "conf": c}
               for k, (w, c) in enumerate([("Build", 0.9), ("a", 0.9), ("bird", 0.5), ("house", 0.9), ("today.", 0.9)])]
    words, _ = align.align(primary, "Build a bed house right today.")
    return words


def test_disputes_are_found_and_the_readings_are_shuffled_but_mapped_back():
    words = disputed()
    found = adjudicate.disputes(words)
    assert [(d["kind"], d["a"], d["b"]) for d in found] == [("alts", "bird", "bed"), ("after", "", "right")]
    assert found[0]["before"] == "Build a" and found[0]["after"] == "house today."
    assert adjudicate.clip_bounds(found[0]) == (0.0, 4.4)
    for index in range(40):  # whichever way round the readings were shown, the pick means the same
        spot = {**found[0], "index": index}
        prompt = adjudicate.build_prompt(spot, ["en"])
        shown_a = prompt.split('Reading A: "')[1].split('"')[0]
        answer = adjudicate.parse_answer(json.dumps({"pick": "A", "heard": shown_a}), spot)
        assert answer["pick"] == (0 if shown_a == "bird" else 1)
    assert {adjudicate._swapped(i, "alts") for i in range(40)} == {True, False}
    assert "(nothing)" in adjudicate.build_prompt(found[1], ["fi", "en"]) and "Finnish, English" in adjudicate.build_prompt(found[1], ["fi", "en"])


def test_answers_are_read_strictly_and_rendered_as_a_star():
    words = disputed()
    spot = adjudicate.disputes(words)[0]
    assert adjudicate.parse_answer("no json here", spot) is None
    assert adjudicate.parse_answer('{"pick": "maybe"}', spot) is None
    assert adjudicate.parse_answer('```json\n{"pick": "unclear", "heard": ""}\n```', spot) == {"pick": "unclear"}
    assert adjudicate.parse_answer('{"pick": "other", "heard": "beard"}', spot) == {"pick": "other", "heard": "beard"}
    render = lambda: " ".join(notation.render_blocks(words, 0.7))  # noqa: E731
    assert render() == "Build a {bird|bed} house {∅|right} today."
    words[2]["adj"] = {"pick": 1}
    words[3]["adj_after"] = {"pick": 0}
    assert render() == "Build a {bird|bed*} house {∅*|right} today."
    words[2]["adj"] = {"pick": "other", "heard": "beard"}
    assert render() == "Build a {bird|bed|beard*} house {∅*|right} today."
    words[2]["adj"] = {"pick": "unclear"}
    assert render().startswith("Build a {bird|bed} house")
    assert adjudicate.tally(words) == {"judged": 2, "primary": 1, "second": 0, "other": 0, "unclear": 1}
    assert all("adj" not in w and "adj_after" not in w for w in align.strip_reader(words))  # a new reading starts clean


def test_select_takes_the_most_doubtful_spots_and_returns_them_in_order():
    found = [{"index": k, "start_ms": k * 100, "conf": c} for k, c in enumerate([0.9, None, 0.2, 0.6, 0.4])]
    assert [d["index"] for d in adjudicate.select(found, 3)] == [2, 3, 4]
    assert [d["index"] for d in adjudicate.select(found, 0)] == [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------- exports

def transcript():
    words = disputed()
    for word in words:
        word["speaker"] = "1"
    words += [{"text": "Yes", "start_ms": 6000, "end_ms": 6300, "conf": 0.3, "speaker": "2"},
              {"text": "<ok>", "start_ms": 6300, "end_ms": 6600, "conf": 0.9, "speaker": "2"}]
    words[2]["adj"] = {"pick": 1}
    return {"schema": "voicememos-mcp/transcript@1",
            "memo": {"id": "x", "title": "Plans & more", "recorded": "2026-11-26T10:00:00+02:00", "duration_s": 7},
            "engine": {"provider": "soniox", "model": "stt-async-v6"},
            "readers": [{"role": "second", "model": "gemini-4-argon"}, {"role": "adjudicator", "model": "gemini-4-neon"}],
            "speakers": {"1": {"label": "Alex", "share": 0.7}, "2": {"label": "S2", "share": 0.3}},
            "redactions": [{"start_ms": 3000, "end_ms": 5000, "date": "2026-11-21"}], "words": words}


def test_webvtt_has_cues_with_voices_times_and_marks():
    vtt = export.webvtt(transcript(), 0.7)
    lines = vtt.split("\n")
    assert lines[0] == "WEBVTT" and "NOTE Plans & more" in lines[2]
    assert "00:00:00.000 --> 00:00:02.400" in vtt and "<v Alex>Build a {bird|bed*} house {∅|right} today." in vtt
    assert "00:00:03.000 --> 00:00:05.000\n[poistettu 21.11.2026: 0:03–0:05]" in vtt
    assert "<v S2>Yes (?) &lt;ok&gt;" in vtt


def test_tei_is_well_formed_and_carries_readings_confidence_and_the_gap():
    from xml.etree import ElementTree

    xml = export.tei(transcript(), 0.7)
    root = ElementTree.fromstring(xml)
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    turns = root.findall(".//t:u", ns)
    assert [u.get("who") for u in turns] == ["#spk1", "#spk2"]
    whens = {w.get("{http://www.w3.org/XML/1998/namespace}id"): w.get("absolute") for w in root.findall(".//t:when", ns)}
    assert whens[turns[0].get("start")[1:]] == "0s" and whens[turns[1].get("end")[1:]] == "6.6s"
    readings = [(s.get("resp"), s.text, s.get("ana")) for s in turns[0].find("t:choice", ns)]
    assert readings == [("#primary", "bird", None), ("#second", "bed", "#picked")]
    assert turns[1].find("t:unclear", ns).get("cert") == "0.30" and turns[1].findall(".//t:w", ns)[1].text == "<ok>"
    assert root.find(".//t:gap", ns).get("reason") == "redacted" and root.find(".//t:title", ns).text == "Plans & more"
    assert [p.find("t:persName", ns).text for p in root.findall(".//t:person", ns)] == ["Alex", "S2"]


# ----------------------------------------------------------------------------- through the worker

CONVERSATION = [
    {"text": " Hello", "start_ms": 0, "end_ms": 900, "confidence": 0.99, "speaker": "1"},
    {"text": " there", "start_ms": 1000, "end_ms": 1900, "confidence": 0.5, "speaker": "1"},
    {"text": " friend", "start_ms": 2000, "end_ms": 7900, "confidence": 0.99, "speaker": "1"},
    {"text": " Hi", "start_ms": 8000, "end_ms": 8900, "confidence": 0.99, "speaker": "2"},
    {"text": " back", "start_ms": 9000, "end_ms": 15900, "confidence": 0.99, "speaker": "2"},
]


def test_passes_over_an_existing_transcript_through_the_queue(cfg, library):
    audio = np.concatenate([tone(OWNER, 8, seed=5), tone(OTHER, 8, seed=6)])
    reader = lambda path: {"text": "Hello the friend hi back", "model": "gemini-4-argon",  # noqa: E731
                           "usage": {"totalTokenCount": 10}, "finish_reason": "STOP"}
    asked = []

    def judge(clip, prompt):
        asked.append(prompt)
        return {"text": '{"pick": "B", "heard": "x"}', "model": "gemini-4-neon", "usage": {"totalTokenCount": 7}}

    worker = lambda: Worker(cfg, transcriber=FakeTranscriber(tokens=CONVERSATION), second_reader=reader,  # noqa: E731
                            embedder=fake_embed, audio_loader=lambda p: audio, adjudicator=judge,
                            audio_cutter=lambda path, a, b, out: out.write_bytes(b"clip") and out or out)
    worker().run(only=["aaaa1111"])
    path = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json"))
    before = json.loads(path.read_text())
    assert "prosody" not in before and "{there|the}" in path.with_suffix(".md").read_text() and not asked

    tools = Tools(cfg)
    with pytest.raises(ToolError):
        tools.annotate(["loudness"], memos=["aaaa1111"])
    with pytest.raises(ToolError):
        tools.annotate(["prosody"])
    job = tools.annotate(["prosody", "adjudicate", "speakers"], memos=["aaaa1111"])
    worker().run()
    item = tools.transcribe_status(job["job_id"])["items"][0]
    assert item["status"] == "annotated" and item["done"]["prosody"] == {"measured_words": 5}
    assert item["done"]["adjudicate"]["judged"] == 1 and len(asked) == 1
    after = json.loads(path.read_text())
    assert all("pros" in w for w in after["words"]) and set(after["prosody"]["speakers"]) == {"1", "2"}
    third = after["readers"][-1]
    assert third["role"] == "adjudicator" and third["model"] == "gemini-4-neon" and third["judged"] == 1
    md = path.with_suffix(".md").read_text()
    assert ("{there*|the}" in md or "{there|the*}" in md) and "Third listening (gemini-4-neon) of 1 of the 1 spots" in md
    assert after["speaker_split"]["changed_words"] == 0 and "divided by voice" not in md
    listed = tools.list_memos(title="Morning")["memos"][0]
    assert listed["annotations"]["prosody"] is True and listed["annotations"]["adjudicated"]["judged"] == 1
    usage = next(iter(tools.status()["usage"].values()))
    # Two second readings (the queue's run also transcribed the other memo) and one third listening.
    assert usage["gemini_calls"] == 3 and usage["gemini_tokens"] == 27

    # Done passes are not done again without redo; the view of the measurements reads from the data.
    job = tools.annotate(["adjudicate", "prosody"], memos=["aaaa1111"])
    worker().run()
    assert tools.transcribe_status(job["job_id"])["items"][0]["done"] == {"adjudicate": "already done",
                                                                         "prosody": "already done"}
    assert len(asked) == 1
    view = tools.read("aaaa1111", format="ca")["text"]
    assert "Conversation-analysis view" in view and "[00:00] S1: Hello" in view and "S2: " in view
    assert tools.read("aaaa1111", format="vtt")["text"].startswith("WEBVTT")
    assert tools.read("aaaa1111", format="tei")["text"].startswith("<?xml")


def test_new_memos_go_through_the_passes_that_are_switched_on(cfg, library):
    from voicememos_mcp.config import load_config

    on = load_config(path=cfg.data_dir / "missing.env",
                     env={**cfg.raw, "VOICEMEMOS_PROSODY": "1", "VOICEMEMOS_SPEAKER_SPLIT": "1",
                          "VOICEMEMOS_QUARANTINE_S": "5"})
    audio = np.concatenate([tone(OWNER, 8, seed=5), tone(OTHER, 8, seed=6)])
    Worker(on, transcriber=FakeTranscriber(tokens=CONVERSATION), embedder=fake_embed,
           audio_loader=lambda p: audio).run(only=["aaaa1111"])
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["prosody"]["measured_words"] == 5 and data["speaker_split"]["speakers"] == 2
    assert "quarantine" not in data  # no voiceprint, so nobody is unknown beside the owner
    # A dictation (one speaker) is never divided, whatever the room does to the voice.
    solo = [dict(t, speaker="1") for t in CONVERSATION]
    Worker(on, transcriber=FakeTranscriber(tokens=solo), embedder=fake_embed,
           audio_loader=lambda p: audio).run(only=["bbbb2222"])
    data = json.loads(next(cfg.output_dir.rglob("*car-notes_bbbb2222.json")).read_text())
    assert "speaker_split" not in data and data["speakers"] == {}


def test_a_voice_from_a_file_is_stored_without_a_memo(cfg, library, tmp_path):
    Worker(cfg, transcriber=FakeTranscriber()).run()
    sample = tmp_path / "tts.m4a"
    sample.write_bytes(b"x")
    worker = Worker(cfg, transcriber=FakeTranscriber(), embedder=fake_embed, audio_loader=lambda p: tone(THIRD, 20, seed=9))
    with pytest.raises(RuntimeError):
        worker.add_other_voice_file(sample, "Reader")
    vp = enrolled()
    vp["model"] = cfg.get("VOICEMEMOS_VOICEPRINT_MODEL")
    worker.store.voiceprint_path.write_text(json.dumps(vp))
    summary = worker.add_other_voice_file(sample, "Reader")
    assert worker.add_other_voice_file(sample, "Reader") == summary  # the same file and name replace
    stored = json.loads(worker.store.voiceprint_path.read_text())["others"]
    assert len(stored) == 1 and stored[0]["file"] == "tts.m4a" and "show" not in stored[0]
    with pytest.raises(RuntimeError):
        worker.add_other_voice_file(sample, "Reader", show=True)
