"""Measurements per word and the conversation-analysis view, with synthetic audio."""
import pytest

np = pytest.importorskip("numpy")

from voicememos_mcp import ca, prosody  # noqa: E402

SR = prosody.SAMPLE_RATE


def voice(f0, seconds, amplitude=0.2):
    """A voiced sound: a fundamental with three harmonics."""
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    return (amplitude * wave / 2).astype(np.float32)


def word(text, start_ms, end_ms, speaker="1", **extra):
    return {"text": text, "start_ms": start_ms, "end_ms": end_ms, "conf": 0.99, "speaker": speaker, **extra}


def test_pitch_of_a_frame_is_the_fundamental_not_the_octave_below():
    for f0 in (90, 120, 210, 330):
        level, pitch = prosody.frame_features(voice(f0, 0.5))
        assert abs(np.nanmedian(pitch) - f0) / f0 < 0.03
    level, pitch = prosody.frame_features(np.zeros(SR, dtype=np.float32))
    assert np.isnan(pitch).all() and level.max() < -100


def test_syllables():
    assert [prosody.syllables(w) for w in ("talo", "joo", "aio", "strength", "ajattelin,", "5", "")] == [2, 1, 2, 1, 4, 0, 0]


def test_words_are_measured_against_the_speakers_own_level():
    # Speaker 1: five usual words at 120 Hz, one loud and high. Speaker 2 speaks higher and more quietly throughout.
    audio = np.concatenate([voice(120, 0.5)] * 5 + [voice(180, 0.5, amplitude=0.4)] + [voice(220, 0.5, amplitude=0.05)] * 4)
    words = [word("talo", k * 500, k * 500 + 500) for k in range(6)] + \
            [word("talo", 3000 + k * 500, 3500 + k * 500, speaker="2") for k in range(4)]
    out, summary = prosody.measure(audio, words)
    assert summary["measured_words"] == 10 and set(summary["speakers"]) == {"1", "2"}
    assert abs(summary["speakers"]["1"]["f0_hz"] - 120) < 3 and abs(summary["speakers"]["2"]["f0_hz"] - 220) < 5
    usual, raised = out[0]["pros"], out[5]["pros"]
    assert abs(usual["db"]) < 0.5 and abs(usual["st"]) < 0.3 and usual["rate"] == 1.0
    assert 5.5 < raised["db"] < 6.5 and 6.5 < raised["st"] < 7.5  # twice the amplitude, a fifth up
    assert abs(out[7]["pros"]["db"]) < 0.5 and abs(out[7]["pros"]["st"]) < 0.3  # quiet and high is speaker 2's usual
    assert "pros" not in words[0]  # the input is not changed


def test_words_without_times_are_left_alone():
    out, summary = prosody.measure(voice(120, 1.0), [word("a", None, None), word("talo", 0, 500), word("talo", 500, 1000)])
    assert "pros" not in out[0] and out[1]["pros"]["db"] == 0.0 and summary["measured_words"] == 2


def test_ca_view_marks_pauses_latching_and_what_was_measured():
    words = [
        word("no", 0, 300), word("niin", 350, 700),
        word("mä", 1500, 1700, pros={"db": 0.0, "st": 0.0, "rate": 1.0}),
        word("ajattelin", 1700, 2300, pros={"db": 1.0, "st": 0.5, "rate": 1.0}),
        word("että", 2300, 2500, pros={"db": 0.0, "st": -0.5, "rate": 1.0}),
        word("EI", 2500, 2800, pros={"db": 12.0, "st": 7.5, "rate": 1.0}),
        word("käy", 2800, 3100, pros={"db": 10.5, "st": 1.0, "rate": 1.0}),
        word("joo", 3150, 4200, speaker="2", pros={"db": 0.0, "st": 0.0, "rate": 0.2}),
        word("se", 4600, 4700, speaker="2", pros={"rate": 2.0}), word("on", 4700, 4800, speaker="2", pros={"rate": 2.1}),
        word("ihan", 4800, 4900, speaker="2", pros={"rate": 1.9}), word("ok.", 4900, 5200, speaker="2", pros={"rate": 1.0}),
    ]
    speakers = {"1": {"label": "Ville"}, "2": {"label": "S2"}}
    lines = ca.render(words, 0.7, speakers)
    assert lines == ["[00:00] Ville: no niin (0.8) mä ajattelin että ↑EI KÄY=",
                     "[00:03] S2: = joo:: (.) >se on ihan< ok."]
    counts = ca.mark_counts(words)
    assert counts["loud"] == 2 and counts["up"] == 1 and counts["fast"] == 3 and counts["stretch"] == 1


def test_ca_view_marks_single_loud_words_only_in_runs_and_keeps_uncertainty_marks():
    words = [word("a", 0, 200, pros={"db": 12.0}), word("b", 200, 400, pros={"db": 0.0}),
             word("c", 400, 600, conf=0.2, pros={"db": 12.0}), word("d", 600, 800, alts=["e"], adj={"pick": 1})]
    assert ca.render(words, 0.7) == ["[00:00] a b c (?) {d|e*}"]
    assert "loud" not in ca.mark_counts(words)
