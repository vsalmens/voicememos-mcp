"""The own speaker split with synthetic voices: a direction per person plus noise per unit."""
import pytest

np = pytest.importorskip("numpy")

from voicememos_mcp import diarize  # noqa: E402

DIM = 64


def person(seed):
    v = np.random.default_rng(1000 + seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def talk(plan, noise=0.75, seed=0):
    """plan: [(engine speaker, person, units)] -> (words, units with vectors). Each unit is 4 s of
    one-second words; a unit's vector is the person's direction plus noise, like a real embedding
    of a few seconds (similarity to the person's own average about 0.8)."""
    rng = np.random.default_rng(seed)
    words, who = [], []
    t = 0
    for speaker, who_speaks, count in plan:
        for _ in range(count):
            for k in range(4):
                words.append({"text": "w", "start_ms": t + k * 1000, "end_ms": t + k * 1000 + 900, "speaker": speaker})
            who.append(who_speaks)
            t += 4500  # a pause longer than UNIT_GAP_MS ends the unit
    found = diarize.units(words)
    assert len(found) == len(who)
    for unit, who_speaks in zip(found, who):
        vector = person(who_speaks) + noise * rng.standard_normal(DIM).astype(np.float32) / np.sqrt(DIM)
        unit["vector"] = vector / np.linalg.norm(vector)
    return words, found, who


def people_of(words, found, who):
    """speaker key -> the set of people whose units ended under it."""
    out = {}
    for unit, who_speaks in zip(found, who):
        out.setdefault(words[unit["first"]]["speaker"], set()).add(who_speaks)
    return out


def test_units_end_at_pauses_speaker_changes_and_a_maximum_length():
    words = [{"text": "w", "start_ms": k * 1000, "end_ms": k * 1000 + 900, "speaker": "1"} for k in range(10)]
    words[3]["speaker"] = "2"
    spans = [(u["speaker"], u["first"], u["last"]) for u in diarize.units(words)]
    assert spans == [("1", 0, 3), ("2", 3, 4), ("1", 4, 10)]
    long = [{"text": "w", "start_ms": k * 1000, "end_ms": k * 1000 + 950, "speaker": "1"} for k in range(20)]
    assert all(u["end_s"] - u["start_s"] <= diarize.UNIT_MAX_S for u in diarize.units(long))


def test_a_recording_the_engine_got_right_is_unchanged():
    words, found, who = talk([("1", 1, 20), ("2", 2, 20), ("3", 3, 12)])
    new, summary = diarize.regroup(words, found)
    assert new == words and summary["changed_words"] == 0 and summary["speakers"] == 3 and "result" not in summary


def test_two_people_under_one_number_are_divided():
    words, found, who = talk([("1", 1, 20), ("1", 2, 15), ("2", 3, 20)])
    new, summary = diarize.regroup(words, found)
    assert summary["speakers"] == 3 and summary["engine_speakers"] == 2
    assert all(len(v) == 1 for v in people_of(new, found, who).values())
    moved = [w for w in new if "speaker_engine" in w]
    assert len(moved) == 15 * 4 and {w["speaker_engine"] for w in moved} == {"1"} and {w["speaker"] for w in moved} == {"3"}
    assert summary["result"]["divided"][0]["engine_speaker"] == "1"
    # Running it again on its own result starts from the engine's speakers and gives the same.
    again, _ = diarize.regroup(diarize.restore(new), found)
    assert again == new


def test_one_person_under_two_numbers_is_joined_and_talk_under_a_wrong_number_moves():
    # Person 1 is the engine's speakers 1 and 3; a part of person 2's talk is under speaker 1 as well.
    words, found, who = talk([("1", 1, 20), ("2", 2, 20), ("3", 1, 15), ("1", 2, 6)])
    new, summary = diarize.regroup(words, found)
    after = people_of(new, found, who)
    assert summary["speakers"] == 2 and all(len(v) == 1 for v in after.values())
    assert summary["result"]["joined"] and all(s >= diarize.SIMILARITY for s in summary["result"]["joined"])


def test_a_small_side_is_divided_off_only_when_it_is_very_different():
    assert diarize.different(0.50, 15, 300) and not diarize.different(0.60, 15, 300)
    assert diarize.different(0.60, 60, 300) and not diarize.different(0.75, 60, 300)
    assert diarize.same(0.75, 60, 300) and not diarize.same(0.75, 15, 300) and diarize.same(0.85, 15, 300)


def test_parts_of_a_long_recording_are_done_each_on_their_own():
    a_words, a_found, _ = talk([("1:1", 1, 20), ("1:1", 2, 15)])
    b_words, b_found, _ = talk([("2:1", 1, 20), ("2:2", 2, 20)], seed=1)
    for unit in b_found:
        unit["first"] += len(a_words)
        unit["last"] += len(a_words)
    new, summary = diarize.regroup(a_words + b_words, a_found + b_found)
    keys = {w["speaker"] for w in new}
    assert keys == {"1:1", "1:2", "2:1", "2:2"} and summary["parts"][0]["part"] == 1 and len(summary["parts"]) == 1


def test_split_embeds_from_audio():
    sr = 16000
    tones = {1: 200, 2: 900}

    def embed(samples):
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1 / sr)
        return np.array([spectrum[(freqs >= a) & (freqs < b)].sum() for a, b in ((100, 400), (400, 700), (700, 1100))]
                        + [1.0], dtype=np.float32)

    plan = [("1", 1, 10), ("1", 2, 8), ("2", 2, 6)]
    words, _, who = talk(plan)
    audio = np.zeros(int(words[-1]["end_ms"] / 1000 * sr) + sr, dtype=np.float32)
    for unit, who_speaks in zip(diarize.units(words), who):
        a, b = int(unit["start_s"] * sr), int(unit["end_s"] * sr)
        audio[a:b] = 0.5 * np.sin(2 * np.pi * tones[who_speaks] * np.arange(b - a) / sr)
    new, summary = diarize.split(audio, words, embed)
    assert summary["speakers"] == 2 and summary["changed_words"] == 8 * 4
