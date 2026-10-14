from voicememos_mcp import notation


def w(text, start_s, conf=0.95, speaker=None):
    item = {"text": text, "start_ms": int(start_s * 1000), "end_ms": int(start_s * 1000) + 400, "conf": conf}
    if speaker:
        item["speaker"] = speaker
    return item


def test_repetition_loop_is_collapsed_and_marked():
    words = [w("se", 0), w("oli", 0.5)] + [w("niinku", 1 + k * 0.5) for k in range(7)] + [w("tosi", 5), w("hyvää.", 5.5)]
    text = " ".join(notation.render_blocks(words, 0.7))
    assert text == "se oli [?: niinku ×7] tosi hyvää."
    flags = [x.get("flag") for x in notation.annotate(words, 0.7)]
    assert flags.count("repetition") == 7
    assert notation.stats(words, 0.7)["uncertain"] == 7


def test_short_repetition_and_other_speakers_are_kept():
    words = [w("tosi", 0), w("tosi", 0.5), w("tosi", 1), w("hyvä", 1.5)]
    assert " ".join(notation.render_blocks(words, 0.7)) == "tosi tosi tosi hyvä"
    mixed = [w("joo", 0, speaker="1"), w("joo", 0.5, speaker="1"), w("joo", 1, speaker="2"), w("joo", 1.5, speaker="2")]
    assert notation.repetition_runs(mixed) == {}


def test_time_stamps_every_minute():
    words = [w("alku", 0), w("toinen", 70), w("kolmas", 100), w("neljäs", 140)]
    blocks = notation.render_blocks(words, 0.7, pause_ms=2500, stamp_every_s=60)
    assert blocks == ["alku", "[01:10] toinen", "kolmas", "[02:20] neljäs"]
    long = [w("a", 0), w("b", 3700)]
    assert notation.render_blocks(long, 0.7, stamp_every_s=60)[1] == "[1:01:40] b"
    assert notation.render_blocks(words, 0.7, stamp_every_s=0)[1] == "toinen"


def test_time_stamp_precedes_speaker_label():
    speakers = {"1": {"label": "S1"}, "2": {"label": "S2"}}
    words = [w("hei", 0, speaker="1"), w("moi", 65, speaker="2")]
    assert notation.render_blocks(words, 0.7, speakers=speakers, stamp_every_s=60)[1] == "[01:05] **S2:** moi"
