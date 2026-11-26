"""The owner's tone against their own baseline: only the owner's voice, deviations and not raw
figures, a side file and one header line, and a transcript text that does not change (synthetic
tones and a fake embedder; one test runs Praat itself on a synthetic signal)."""
import json

import pytest

np = pytest.importorskip("numpy")

from tests.conftest import FakeTranscriber  # noqa: E402
from tests.test_voiceprint import OTHER, OWNER, SR, fake_embed, tone as sine  # noqa: E402
from voicememos_mcp import tone, voiceprint  # noqa: E402
from voicememos_mcp.config import load_config  # noqa: E402
from voicememos_mcp.worker import Worker  # noqa: E402
def _enrolled(cfg, extra_env=None):
    """A config with the owner's voiceprint enrolled from four synthetic memos."""
    env = dict(cfg.raw, VOICEMEMOS_VOICEPRINT_MODEL="test-model", **(extra_env or {}))
    cfg2 = load_config(path=cfg.data_dir / "missing.env", env=env)
    cfg2.data_dir.mkdir(parents=True, exist_ok=True)
    memos = {f"m{i}": sine(OWNER + i * 3, 20, seed=i, noise=0.05 + 0.02 * i) for i in range(4)}
    voiceprint.Voiceprint(cfg2.data_dir / "voiceprint.json").save(voiceprint.enroll("Alex", "test-model", memos, fake_embed))
    return cfg2


DICTATION = [{"text": " sana", "start_ms": i * 1000, "end_ms": i * 1000 + 900, "confidence": 0.99} for i in range(14)]
FIGURES = {"pitch": 200.0, "pitch_var": 2.0, "level": 60.0, "jitter": 0.012, "shimmer": 0.03}


def measurer(samples, spans):
    assert spans and all(b > a for a, b in spans)
    return dict(FIGURES)


def fill_baseline(cfg, n=6):
    history = tone.History(cfg.data_dir / "tone.json")
    for i in range(n):
        history.put(f"OLD{i}", f"2020-01-{i + 1:02d}T10:00:00+02:00",
                    {"rate": 4.0 + 0.1 * i, "pause_share": 0.2 + 0.02 * i, "pitch": 160.0 + 2 * i,
                     "pitch_var": 1.8 + 0.08 * i, "level": 58.0 + 0.8 * i, "jitter": 0.010 + 0.001 * i,
                     "shimmer": 0.028 + 0.001 * i})
    return history


def run(cfg, audio, tokens=DICTATION, memo="aaaa1111", redo=False, with_measurer=measurer):
    worker = Worker(cfg, transcriber=FakeTranscriber(tokens=tokens), embedder=fake_embed, audio_loader=lambda p: audio)
    worker.tone_measurer = with_measurer
    worker.run(only=[memo], redo=redo)
    return worker


def files(cfg, pattern="*morning-walk_aaaa1111"):
    md = next(cfg.output_dir.rglob(pattern + ".md")).read_text()
    data = json.loads(next(p for p in cfg.output_dir.rglob(pattern + ".json") if not p.name.endswith(".tone.json")).read_text())
    side = list(cfg.output_dir.rglob(pattern + ".tone.json"))
    return md, data, side


def test_rate_and_pauses_come_from_the_word_times_of_the_owners_own_turns():
    words = ([{"text": "talo", "start_ms": i * 1000, "end_ms": i * 1000 + 500, "speaker": "1"} for i in range(12)]
             + [{"text": "joo", "start_ms": 12000, "end_ms": 12400, "speaker": "2"}]
             + [{"text": "auto", "start_ms": 13000 + i * 1000, "end_ms": 13500 + i * 1000, "speaker": "1"} for i in range(12)])
    figures = tone.from_times(words, "1")
    # 24 words of two syllables in 12 s of speaking; 22 pauses of 500 ms, and the gap across the other speaker's word is none
    assert figures == {"rate": 4.0, "pause_share": round(11000 / 23000, 4)}
    assert tone.from_times(words, "2") == {}  # under ten seconds of speech: not measured
    assert tone.from_times([{"text": "a", "start_ms": 0, "end_ms": 11000}, {"text": "b", "start_ms": 60000, "end_ms": 61000}], None)["pause_share"] == 0.0  # a minute's gap is not a pause


def test_figures_are_told_as_deviations_from_the_owners_own_median_and_the_line_names_what_stands_out():
    earlier = [{"raw": {"rate": r, "pitch": p}} for r, p in zip((4.0, 4.1, 4.2, 4.3, 4.4), (160, 162, 164, 166, 168))]
    dev, n = tone.deviations({"rate": 4.2, "pitch": 190.0, "level": 60.0}, earlier)
    assert n == 5 and dev["rate"] == 0.0 and dev["pitch"] > 5 and "level" not in dev  # the baseline has no level
    assert tone.labels(dev) == ["korkea"] and tone.line(dev) == "[sävy: korkea]"
    assert tone.line({"rate": 0.2, "pitch": -0.4}) == "[sävy: tavallinen]"
    assert tone.line({"rate": 1.5, "pitch": 2.0, "pause_share": -1.2, "jitter": 9.0}) == "[sävy: nopea, korkea, vähän taukoja]"
    assert tone.deviations({"rate": 4.2}, earlier[:4]) == ({}, 4)  # four recordings are not a baseline
    assert tone.line({}) is None and tone.line({"jitter": 3.0}) is None
    flat = [{"raw": {"rate": 4.0}}] * 6
    assert tone.deviations({"rate": 5.0}, flat) == ({}, 6)  # a baseline that does not vary measures nothing


def test_only_a_firm_voiceprint_match_is_the_owner():
    assert tone.owner_speaker({}, {"result": "match"}) == (True, None)
    assert tone.owner_speaker({}, {"result": "possible"}) == (False, None)
    assert tone.owner_speaker({}, None) == (False, None)
    speakers = {"1": {"label": "Alex", "basis": "voiceprint"}, "2": {"label": "S2"}}
    assert tone.owner_speaker(speakers, {"result": "match", "speaker": "1"}) == (True, "1")
    assert tone.owner_speaker(speakers, {"result": "possible", "speaker": "1"}) == (False, None)
    assert tone.owner_speaker(speakers, {"result": "no match"}) == (False, None)


def test_praat_measures_a_synthetic_voice():
    pytest.importorskip("parselmouth")

    def voiced(freq):
        t = np.arange(SR * 12) / SR
        return (0.3 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)).astype(np.float32)

    low, high = tone.acoustic(voiced(150), [(0.0, 12.0)]), tone.acoustic(voiced(220), [(0.0, 12.0)])
    assert abs(low["pitch"] - 150) < 2 and abs(high["pitch"] - 220) < 2
    assert set(low) == {"pitch", "pitch_var", "level", "jitter", "shimmer"} and low["pitch_var"] < 0.2 and low["jitter"] < 0.01
    assert tone.acoustic(voiced(150), [(0.0, 1.0)]) == {}  # too little to measure
    assert tone.acoustic(voiced(150), [(0.0, 12.0)]) == low  # the same audio, the same figures


def test_the_owners_dictation_gets_a_side_file_and_a_header_line_and_the_same_run_twice_the_same_result(cfg, library):
    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})
    fill_baseline(on)
    run(on, sine(OWNER, 20, seed=5))
    md, data, side = files(on)
    assert "> [sävy: hidas, korkea, vähän taukoja]" in md.split("## Transcript")[0]
    assert data["tone"] == {"line": "[sävy: hidas, korkea, vähän taukoja]", "labels": ["hidas", "korkea", "vähän taukoja"],
                            "baseline": {"recordings": 6, "of": 30, "needed": 5}}
    content = json.loads(side[0].read_text())
    assert content["memo_id"].startswith("AAAA1111") and content["line"] == data["tone"]["line"]
    assert set(content["deviations"]) == set(tone.MEASURES) and content["deviations"]["pitch"] > 1
    text = side[0].read_text()
    assert "200.0" not in text and "sana" not in text and "Morning walk" not in text  # deviations: no raw figure, no word, no title
    assert data["timing"]["extras_s"].keys() >= {"tone"}
    stored = json.loads((on.data_dir / "tone.json").read_text())["memos"]
    assert stored[content["memo_id"]]["raw"]["pitch"] == 200.0 and oct((on.data_dir / "tone.json").stat().st_mode & 0o777) == "0o600"

    # again, after a later recording has joined the history: the baseline is of earlier recordings only
    tone.History(on.data_dir / "tone.json").put("LATER", "2999-01-01T00:00:00+02:00", {"rate": 9.0, "pitch": 400.0})
    run(on, sine(OWNER, 20, seed=5), redo=True)
    assert files(on)[2][0].read_text() == text


def test_the_transcript_text_is_byte_for_byte_the_same_with_the_layer_on_and_off(cfg, library):
    off = _enrolled(cfg)
    run(off, sine(OWNER, 20, seed=5))
    md_off, data_off, side_off = files(off)
    assert not side_off and "sävy" not in md_off and "tone" not in data_off and not (off.data_dir / "tone.json").exists()

    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})
    fill_baseline(on)
    run(on, sine(OWNER, 20, seed=5), redo=True)
    md_on, data_on, side_on = files(on)
    assert side_on and "sävy" in md_on
    assert md_on.split("## Transcript")[1].encode() == md_off.split("## Transcript")[1].encode()
    assert data_on["words"] == data_off["words"] and data_on["source_tokens"] == data_off["source_tokens"]


def test_a_recording_without_the_owners_voice_gets_no_tone_at_all(cfg, library):
    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})
    fill_baseline(on)
    run(on, sine(OWNER, 20, seed=5))
    assert files(on)[2]
    # the same recording transcribed again, now someone else's voice: the earlier side file and baseline row go
    run(on, sine(OTHER, 20, seed=6), redo=True)
    md, data, side = files(on)
    assert not side and "sävy" not in md and "tone" not in data
    assert not any(key.startswith("AAAA1111") for key in json.loads((on.data_dir / "tone.json").read_text())["memos"])

    # two speakers, the owner among them: only the owner's stretches reach the measurer
    conversation = ([{"text": " sana", "start_ms": i * 1000, "end_ms": i * 1000 + 900, "confidence": 0.99, "speaker": "1"} for i in range(12)]
                    + [{"text": " muu", "start_ms": 12000 + i * 1000, "end_ms": 12900 + i * 1000, "confidence": 0.99, "speaker": "2"} for i in range(12)])
    seen = []

    def spans_only(samples, spans):
        seen.extend(spans)
        return dict(FIGURES)

    run(on, np.concatenate([sine(OWNER, 12, seed=5), sine(OTHER, 12, seed=6)]), tokens=conversation, redo=True, with_measurer=spans_only)
    assert seen and all(b <= 12.0 for _, b in seen) and files(on)[2]


def test_a_recording_in_an_excluded_folder_yields_nothing(cfg, library):
    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})  # conftest: the folder "Private" is excluded
    fill_baseline(on)
    worker = Worker(on, transcriber=FakeTranscriber(tokens=DICTATION), embedder=fake_embed, audio_loader=lambda p: sine(OWNER, 20, seed=5))
    measured = []
    worker.tone_measurer = lambda samples, spans: measured.append(1) or dict(FIGURES)
    memo = next(m for m in worker._memos() if m.id.startswith("EEEE5555"))
    assert worker.transcribe_memo(memo, force=True)["status"] == "excluded" and not measured
    worker.run()  # the other recordings are measured, the excluded one never
    stored = json.loads((on.data_dir / "tone.json").read_text())["memos"]
    assert measured and any(key.startswith("AAAA1111") for key in stored)
    assert not any(key.startswith("EEEE5555") for key in stored)
    assert not list(on.output_dir.rglob("*eeee5555*"))


def test_without_praat_the_figures_from_the_word_times_stand_and_the_rest_is_said_missing(cfg, library):
    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})
    fill_baseline(on)

    def no_praat(samples, spans):
        raise ImportError("No module named 'parselmouth'")

    run(on, sine(OWNER, 20, seed=5), with_measurer=no_praat)
    content = json.loads(files(on)[2][0].read_text())
    assert set(content["deviations"]) == {"rate", "pause_share"} and "praat-parselmouth is not installed" in content["missing"][0]
    assert content["line"] == "[sävy: hidas, vähän taukoja]"


def test_before_five_earlier_recordings_there_is_no_line_and_the_side_file_says_why(cfg, library):
    on = _enrolled(cfg, {"VOICEMEMOS_TONE": "1"})
    fill_baseline(on, n=3)
    run(on, sine(OWNER, 20, seed=5))
    md, data, side = files(on)
    content = json.loads(side[0].read_text())
    assert content["line"] is None and content["deviations"] == {} and "3 earlier recording(s)" in content["note"]
    assert "sävy" not in md and data["tone"]["line"] is None
