from voicememos_mcp import notation
from tests.conftest import TOKENS


def words():
    w, had = notation.tokens_to_words(TOKENS)
    assert had
    return w


def test_tokens_join_subwords_and_take_minimum_confidence():
    w = words()
    assert [x["text"] for x in w] == ["Hello", "world.", "Idea:", "build", "a", "bird", "house."]
    world = w[1]
    assert world["conf"] == 0.40  # min of 0.95 and 0.40; punctuation (0.20) ignored
    assert world["start_ms"] == 320 and world["end_ms"] == 660
    assert w[0]["lang"] == "en"


def test_translation_and_control_tokens_are_skipped():
    tokens = [{"text": "Hi", "confidence": 0.9}, {"text": " hello", "translation_status": "translation"},
              {"text": "<end>"}]
    w, _ = notation.tokens_to_words(tokens)
    assert [x["text"] for x in w] == ["Hi"]


def test_missing_confidence_is_reported():
    w, had = notation.tokens_to_words([{"text": "Hi"}, {"text": " there"}])
    assert not had and all(x["conf"] is None for x in w)
    assert notation.render_blocks(w, 0.7) == ["Hi there"]


def test_single_uncertain_word_keeps_punctuation_after_mark():
    blocks = notation.render_blocks(words(), 0.70)
    assert blocks[0] == "Hello world (?)."
    assert blocks[1] == "Idea: build a bird house."  # pause > 2500 ms starts a new block


def test_run_of_uncertain_words_is_grouped():
    w = [{"text": t, "conf": 0.3, "start_ms": i * 100, "end_ms": i * 100 + 90}
         for i, t in enumerate(["one", "two", "three", "four,"])]
    w.append({"text": "five", "conf": 0.99, "start_ms": 500, "end_ms": 600})
    assert notation.render_blocks(w, 0.7, run_min=4) == ["[?: one two three four], five"]
    assert notation.render_blocks(w, 0.7, run_min=5) == ["one (?) two (?) three (?) four (?), five"]


def test_alternatives_and_gap():
    w = [{"text": "their", "alts": ["there"], "conf": None},
         {"text": "", "conf": None}]
    assert notation.render_word(w[0], 0.7) == "{their|there}"
    assert notation.render_word(w[1], 0.7) == "[?]"


def test_threshold_changes_rendering_without_new_data():
    assert "(?)" not in notation.render_blocks(words(), 0.30)[0]
    s = notation.stats(words(), 0.70)
    assert s["words"] == 7 and s["uncertain"] == 1 and s["certainty"] == "moderate"


def test_annotate_sets_basis():
    annotated = notation.annotate(words(), 0.7)
    assert annotated[1]["flag"] == "uncertain" and annotated[1]["basis"] == "asr_confidence"
    assert "flag" not in annotated[0]


def test_markers_with_custom_spoken_words():
    # Marker words are configurable (any language); non-ASCII text must survive.
    text = "Drove home. Idea: visit the café. Note call (?) the garage tomorrow! Task ok"
    found = notation.extract_markers(text, {"idea": "Idea", "note": "Remember", "task": "Task"})
    assert {"label": "Idea", "text": "visit the café"} in found
    assert {"label": "Remember", "text": "call (?) the garage tomorrow"} in found
    assert len(found) == 2  # "Task ok" is too short
