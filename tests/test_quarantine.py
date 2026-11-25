"""A foreign voice with only a few seconds of speech is set aside, and a synthesiser's voice can be
registered as "not me" from a file (synthetic tones and a fake embedder, as in test_voiceprint)."""
import json

import pytest

np = pytest.importorskip("numpy")

from tests.conftest import FakeTranscriber  # noqa: E402
from tests.test_voiceprint import OTHER, OWNER, SR, fake_embed, tone  # noqa: E402
from voicememos_mcp import cli, quarantine, voiceprint  # noqa: E402
from voicememos_mcp.config import load_config  # noqa: E402
from voicememos_mcp.worker import Worker  # noqa: E402

TTS = 650


def word(text, start_s, end_s, speaker):
    return {"text": text, "start_ms": int(start_s * 1000), "end_ms": int(end_s * 1000), "conf": 0.99, "speaker": speaker}


def labelled(owner="1", **others):
    out = {owner: {"label": "Alex", "basis": "voiceprint", "words": 0, "share": 0.0}}
    out.update({key: {"label": label, "words": 0, "share": 0.0} for key, label in others.items()})
    return out


def test_a_voice_with_a_few_seconds_in_all_is_set_aside_and_a_partner_of_short_replies_stays():
    words = ([word(f"o{i}", i, i + 0.9, "1") for i in range(10)]
             + [word("sure", 10, 10.8, "3"), word("thing", 10.9, 11.6, "3")]            # 1.5 s in all: set aside
             + [word(f"o{i}", i, i + 0.9, "1") for i in range(12, 20)]
             + [word("joo", 20 + 3 * i, 20.7 + 3 * i, "2") for i in range(10)]           # 7 s in ten short replies: stays
             + [word("again", 60, 60.5, "3")])                                          # the same voice later: its own snippet
    kept, speakers, record = quarantine.split(words, labelled("1", **{"2": "S2", "3": "S3"}), 5.0)
    assert {w["speaker"] for w in kept} == {"1", "2"} and len(kept) == 28
    assert set(speakers) == {"1", "2"} and speakers["1"]["words"] == 18 and speakers["2"]["share"] == round(10 / 28, 3)
    assert record == {"max_s": 5.0, "voices": 1, "snippets": [
        {"speaker": "S3", "start_ms": 10000, "end_ms": 11600, "words": 2},
        {"speaker": "S3", "start_ms": 60000, "end_ms": 60500, "words": 1}]}
    assert "text" not in json.dumps(record)  # the list holds times, never the words
    line = quarantine.line(record)
    assert line.startswith("2 short snippets of 1 other voice") and "`quarantine`" in line
    assert quarantine.line(None) is None


def test_the_owner_is_never_set_aside_and_without_a_recognised_owner_nothing_is():
    # the owner says one short word in someone else's long talk: the owner stays
    words = [word("joo", 0, 0.5, "1")] + [word(f"t{i}", 1 + i, 1.9 + i, "2") for i in range(20)]
    kept, speakers, record = quarantine.split(words, labelled("1", **{"2": "S2"}), 5.0)
    assert kept == words and record is None and set(speakers) == {"1", "2"}
    # a possible match is the owner too
    maybe = {"1": {"label": "Alex (?)", "basis": "voiceprint"}, "2": {"label": "S2"}}
    kept, _, record = quarantine.split([word("a", 0, 9, "1"), word("b", 9, 10, "2")], maybe, 5.0)
    assert [w["text"] for w in kept] == ["a"] and record["snippets"][0]["speaker"] == "S2"
    # nobody recognised: which voice is foreign is not known
    plain = {"1": {"label": "S1"}, "2": {"label": "S2"}}
    words = [word("a", 0, 9, "1"), word("b", 9, 10, "2")]
    assert quarantine.split(words, plain, 5.0) == (words, plain, None)
    assert quarantine.split(words, {}, 5.0) == (words, {}, None)
    # a speaker whose words have no times: how long they spoke is not known, so they stay
    untimed = [word("a", 0, 9, "1"), {"text": "b", "conf": 0.9, "speaker": "2"}]
    assert quarantine.split(untimed, labelled("1", **{"2": "S2"}), 5.0)[2] is None


def _enrolled(cfg, extra_env=None):
    """A config and a worker factory with the owner's voiceprint enrolled from two memos."""
    env = dict(cfg.raw, VOICEMEMOS_VOICEPRINT_MODEL="test-model", **(extra_env or {}))
    cfg2 = load_config(path=cfg.data_dir / "missing.env", env=env)
    cfg2.data_dir.mkdir(parents=True, exist_ok=True)
    memos = {f"m{i}": tone(OWNER + i * 3, 20, seed=i, noise=0.05 + 0.02 * i) for i in range(4)}
    voiceprint.Voiceprint(cfg2.data_dir / "voiceprint.json").save(voiceprint.enroll("Alex", "test-model", memos, fake_embed))
    return cfg2


CONVERSATION = [
    {"text": " Note", "start_ms": 0, "end_ms": 900, "confidence": 0.99, "speaker": "1"},
    {"text": " to", "start_ms": 1000, "end_ms": 1900, "confidence": 0.99, "speaker": "1"},
    {"text": " self", "start_ms": 2000, "end_ms": 9900, "confidence": 0.99, "speaker": "1"},
    {"text": " Certainly", "start_ms": 10000, "end_ms": 11500, "confidence": 0.99, "speaker": "2"},
    {"text": " here", "start_ms": 11600, "end_ms": 13400, "confidence": 0.99, "speaker": "2"},
    {"text": " buy", "start_ms": 14000, "end_ms": 14900, "confidence": 0.99, "speaker": "1"},
    {"text": " milk", "start_ms": 15000, "end_ms": 22000, "confidence": 0.99, "speaker": "1"},
]
AUDIO = np.concatenate([tone(OWNER, 10, seed=5), tone(TTS, 4, seed=6), tone(OWNER, 8, seed=7)])


def test_the_worker_sets_the_snippet_aside_only_with_the_setting_and_says_so_in_the_transcript(cfg, library):
    off = _enrolled(cfg)
    Worker(off, transcriber=FakeTranscriber(tokens=CONVERSATION), embedder=fake_embed,
           audio_loader=lambda p: AUDIO).run(only=["aaaa1111"])
    md = next(off.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "**S2:** Certainly here" in md and "set aside" not in md  # off by default: the transcript is as before
    assert "quarantine" not in json.loads(next(off.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())

    on = _enrolled(cfg, {"VOICEMEMOS_QUARANTINE_SHORT": "1"})
    Worker(on, transcriber=FakeTranscriber(tokens=CONVERSATION), embedder=fake_embed,
           audio_loader=lambda p: AUDIO).run(only=["aaaa1111"], redo=True)
    md = next(on.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    body = md.split("## Transcript")[1]
    assert "Certainly" not in body and "Note to self" in body and "buy milk" in body
    assert "> 1 short snippet of 1 other voice (under 5 s of speech in the whole recording) was set aside" in md
    data = json.loads(next(on.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["quarantine"]["snippets"] == [{"speaker": "S2", "start_ms": 10000, "end_ms": 13400, "words": 2}]
    assert [w["text"] for w in data["words"]] == ["Note", "to", "self", "buy", "milk"]
    assert any("Certainly" in t["text"] for t in data["source_tokens"])  # the engine's raw record still holds them
    assert list(data["speakers"]) == ["1"] and data["speakers"]["1"]["label"] == "Alex"


def test_a_synthesisers_voice_is_registered_as_not_me_from_a_file_and_blocks_the_owners_name(cfg, library, tmp_path, monkeypatch, capsys):
    cfg2 = _enrolled(cfg)
    sample = tmp_path / "claude-tts.wav"
    sample.write_bytes(b"\0" * 64)
    close = tone(OWNER + 2, 30, seed=9)  # a voice the fake embedder cannot tell from the owner's
    monkeypatch.setattr(cli, "load_config", lambda: cfg2)
    import voicememos_mcp.worker as worker_module

    monkeypatch.setattr(worker_module.Worker, "embedder", lambda self: fake_embed)
    monkeypatch.setattr(worker_module.Worker, "load_audio", lambda self, path: close)
    assert cli.main(["voiceprint", "add-other-file", "--file", str(sample), "--name", "Claude TTS"]) == 0
    assert json.loads(capsys.readouterr().out)["other_voices"] == ["Claude TTS"]
    stored = json.loads((cfg2.data_dir / "voiceprint.json").read_text())["others"]
    assert [(o["name"], o["file"], o["memo_id"]) for o in stored] == [("Claude TTS", "claude-tts.wav", None)]
    assert cli.main(["voiceprint", "add-other-file", "--file", str(sample), "--name", "Claude TTS"]) == 0  # again: replaced, not doubled
    capsys.readouterr()
    vp = json.loads((cfg2.data_dir / "voiceprint.json").read_text())
    assert len(vp["others"]) == 1
    # that voice alone in a memo is no longer the owner
    _, summary = voiceprint.identify(close, [], {}, vp, fake_embed)
    assert summary["result"] == "no match" and summary["closest_other"]["name"] == "Claude TTS"
    assert cli.main(["voiceprint", "add-other-file", "--name", "x"]) == 2  # no file
    assert cli.main(["voiceprint", "add-other-file", "--file", str(tmp_path / "missing.wav"), "--name", "x"]) == 2
