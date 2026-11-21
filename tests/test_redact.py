"""``voicememos-mcp redact``: a stretch of time leaves one recording for good (audio, sidecar, transcript,
state) and does not come back, while everything outside it stays as it was. The recording is made up: a
generated tone as Ogg Opus, and an engine result of three parts with a code word in the middle one."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import FakeTranscriber
from voicememos_mcp import cli, notation, opus, redact
from voicememos_mcp.config import load_config
from voicememos_mcp.store import Store
from voicememos_mcp.worker import Worker

FFMPEG = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.access(p, os.X_OK)), None)
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg to make an Ogg Opus file")

SECRET = "kvartsiluumu"
SECONDS = 90
NAME = "plaud_test.ogg"
STEMS = ("kivi", "lato", "meri", "sumu", "tuli", "vesi", "halla", "routa", "paju")
ENDS = ("katu", "pelto", "silta", "torni", "ranta", "polku", "laakso", "niemi", "lahti", "saari")


def make_tokens() -> list[dict]:
    """Nine paragraphs of ten words (a word every 500 ms, 300 ms long), 10 s apart, two speakers taking
    turns. Paragraphs 3 to 5 (30 s to 60 s) are the middle part; the code word is in each of them."""
    tokens = []
    for k in range(9):
        for i in range(10):
            text = SECRET if (k in (3, 4, 5) and i == 6) else f"{STEMS[k]}{ENDS[i]}"
            start = k * 10_000 + i * 500
            tokens.append({"text": " " + text + ("." if i == 9 else ""), "start_ms": start, "end_ms": start + 300,
                           "confidence": 0.99, "language": "fi", "speaker": str(k % 2 + 1)})
    return tokens


TOKENS = make_tokens()


class OnlyOurs(FakeTranscriber):
    def __call__(self, audio: Path):
        self.calls.append(audio.name)
        return [dict(t) for t in (self.tokens if "plaud_test" in audio.name else __import__("tests.conftest").conftest.TOKENS)]


# Asked: from inside the fourth word of paragraph 3 to inside the fourth word of paragraph 5.
FROM_MS, TO_MS = 31_600, 51_600
START_MS, END_MS = 31_300, 52_000  # the end of word 2 of paragraph 3, the start of word 4 of paragraph 5


@pytest.fixture
def env(tmp_path: Path, library: Path):
    cfg = load_config(path=tmp_path / "missing.env", env={
        "VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
        "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_EXCLUDE_FOLDERS": "private", "VOICEMEMOS_WORKER": "none",
        "VOICEMEMOS_SINCE": "", "VOICEMEMOS_TIMESTAMP_EVERY_S": "15", "VOICEMEMOS_IMPORT_DIR": str(tmp_path / "imports"),
    })
    cfg.import_dir.mkdir(parents=True)
    audio = cfg.import_dir / NAME
    subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}",
                    "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", str(audio)], check=True)
    old = time.time() - 3600
    os.utime(audio, (old, old))
    (cfg.import_dir / (NAME + ".json")).write_text(json.dumps(
        {"title": "Palaveri", "recorded": "2026-11-18T19:40:26+02:00", "duration_s": SECONDS, "folder": "Plaud"}))
    fake = FakeTranscriber(TOKENS)
    fake.__class__ = OnlyOurs  # the library's own memos get the usual short result
    Worker(cfg, transcriber=fake).run()
    store = Store(cfg.data_dir, cfg.output_dir, cfg.folder_dirs)
    store.stamp_every_s = cfg.timestamp_every_s
    memo_id = next(k for k, v in store.read_state().items() if v.get("title") == "Palaveri")
    md = store._resolve(store.read_state()[memo_id]["json"]).with_suffix(".md")
    return type("Env", (), {"cfg": cfg, "store": store, "fake": fake, "memo": memo_id, "md": md, "sidecar": md.with_suffix(".json"),
                            "audio": audio, "original_audio": audio.read_bytes(), "tmp": tmp_path})


def files_with(env, word: str = SECRET) -> list[str]:
    needle = word.encode()
    return sorted(p.name for top in (env.tmp,) for p in top.rglob("*")
                  if p.is_file() and needle in p.read_bytes())


def apply(env, start=START_MS, end=END_MS):
    return redact.apply(env.cfg, env.store, env.memo, start, end, "2026-11-21")


def test_plan_moves_the_bounds_to_word_boundaries_and_changes_nothing(env):
    before = {p: p.read_bytes() for p in (env.md, env.sidecar, env.audio, env.store.state_path)}
    plan = redact.plan(env.cfg, env.store, env.memo, FROM_MS, TO_MS)
    assert (plan["start_ms"], plan["end_ms"]) == (START_MS, END_MS) and not plan["whole"] and not plan["already"]
    assert plan["words"] == 7 + 10 + 4 and plan["words_total"] == 90 and plan["source_tokens"] == 21 and plan["duration_ms"] == SECONDS * 1000
    assert plan["edge"]["after"][:3] == ["vesiranta", "vesipolku", SECRET] and len(plan["edge"]["after"]) == 12  # outside: stays
    assert plan["edge"]["before"][-3:] == ["sumukatu", "sumupelto", "sumusilta"] and len(plan["edge"]["before"]) == 12
    lines = env.md.read_text().split("\n")
    top = lines.index("## Transcript")  # above it the line with the word count changes too, and is named
    assert [i for i in plan["transcript"]["lines_cut"] if i < top] == [next(i for i, x in enumerate(lines) if "90 words" in x)]
    assert [lines[i].split("** ")[1].split()[0] for i in plan["transcript"]["lines_cut"] if i > top] == ["sumukatu", "vesikatu"]
    assert [lines[i].split("** ")[1].split()[0] for i in plan["transcript"]["lines_removed"]] == ["tulikatu"]
    assert plan["audio"]["supported"] and plan["audio"]["format"] == "ogg/opus"
    assert all(p.read_bytes() == data for p, data in before.items()) and not env.store.redactions_path.exists()
    assert redact.plan(env.cfg, env.store, env.memo, 0, 10**7)["whole"]
    with pytest.raises(redact.RedactError, match="no words"):
        redact.plan(env.cfg, env.store, env.memo, 6_000, 9_000)
    with pytest.raises(redact.RedactError, match="no transcript on record"):
        redact.plan(env.cfg, env.store, "ei-ole", 0, 1000)


def test_apply_takes_the_range_out_of_every_form(env):
    old_lines = env.md.read_text().split("\n")
    old = json.loads(env.sidecar.read_text())
    info = opus.info(env.audio)
    assert SECRET in env.md.read_text() and len(files_with(env)) >= 2
    done = apply(env)
    marker = "[poistettu 21.11.2026: 0:31–0:52]"
    assert done["marker"] == marker and not done["already"] and done["counts"]["words"] == 21 and done["counts"]["source_tokens"] == 21

    # The transcript: one line in the range's place, and every line outside it as it was, stamps included.
    lines = env.md.read_text().split("\n")
    assert lines.count(marker) == 1
    body = lambda ls: [x for x in ls[ls.index("## Transcript") + 1:] if x.strip()]  # noqa: E731
    new_body, old_body = body(lines), body(old_lines)
    at = new_body.index(marker)
    assert new_body[:at - 1] == old_body[:3] and new_body[at + 2:] == old_body[6:]
    assert new_body[at - 1] == old_body[3].split(" sumutorni")[0] and old_body[3].startswith("**S2:** sumukatu")
    assert new_body[at + 1] == "**S2:** vesiranta vesipolku kvartsiluumu vesiniemi vesilahti vesisaari."  # after the asked end
    assert [x[:7] for x in new_body if x.startswith("[0")] == ["[00:20]", "[01:00]", "[01:20]"]  # 0:40 went with its paragraph

    # The sidecar: words, raw tokens and nothing else of the range; the other words with their times.
    new = json.loads(env.sidecar.read_text())
    assert len(new["words"]) == 69 and len(new["source_tokens"]) == 69
    kept = [w for w in old["words"] if not START_MS <= w["start_ms"] < END_MS]
    assert [(w["text"], w["start_ms"], w["end_ms"]) for w in new["words"]] == [(w["text"], w["start_ms"], w["end_ms"]) for w in kept]
    assert [(r["start_ms"], r["end_ms"], r["date"]) for r in new["redactions"]] == [(START_MS, END_MS, "2026-11-21")]

    # The audio: as long as before, silent in the range, sounding as before outside it.
    heard = done["audio"]["heard"]
    assert done["audio"]["duration_diff_ms"] < 50 and opus.info(env.audio)["duration_ms"] == info["duration_ms"]
    assert heard["decoded"] and heard["samples_before"] == heard["samples_after"] and heard["rms_in_range"] <= opus.QUIET_RMS
    assert heard["rms_outside_after"] > 1000 and abs(heard["rms_outside_after"] - heard["rms_outside_before"]) < 5
    assert opus.unsilent(env.audio, START_MS, END_MS) == 0 and opus.unsilent(env.audio, 0, START_MS - 100) > 100
    assert env.audio.name == NAME and not list(env.cfg.import_dir.glob(".*tmp"))

    # The state follows the file, the record names the range, and verify agrees.
    state = env.store.read_state()[env.memo]
    assert state["fp"].split(":")[0] == str(env.audio.stat().st_size) and state["stats"]["words"] == 69
    assert env.store.redactions_for(env.memo)[0]["state"] == "done"
    checked = redact.verify(env.cfg, env.store, env.memo)
    assert checked["ok"] and checked["ranges"][0]["words_in_range"] == 0 and checked["ranges"][0]["audio_packets_not_silent"] == 0
    # Words of the range that are nowhere else are in no file; the code word stays only where it is outside the range.
    assert files_with(env, "tulikatu") == [] and files_with(env, "sumuranta") == []
    assert json.dumps(done).count(SECRET) == 0


def test_the_code_word_is_in_no_file_when_the_range_holds_every_use_of_it(env):
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    top = env.md.read_text().split("\n").index("## Transcript")
    assert plan["words"] == 30 and [i for i in plan["transcript"]["lines_cut"] if i > top] == [] and len(plan["transcript"]["lines_removed"]) == 3
    apply(env, plan["start_ms"], plan["end_ms"])
    assert files_with(env) == [] and SECRET not in env.md.read_text()
    # A second run changes nothing: not a byte of any file.
    before = {p: p.read_bytes() for top in (env.tmp,) for p in top.rglob("*")
              if p.is_file() and p.name != "worker.lock"}
    again = apply(env, plan["start_ms"], plan["end_ms"])
    assert again["already"] and again["marker"] == "[poistettu 21.11.2026: 0:24–1:00]"
    assert {p: p.read_bytes() for p in before} == before
    assert redact.plan(env.cfg, env.store, env.memo, plan["start_ms"], plan["end_ms"])["already"]


def test_the_range_does_not_come_back(env):
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    marker = apply(env, plan["start_ms"], plan["end_ms"])["marker"]
    md = env.md.read_text()
    # The next worker runs do not transcribe it again (the file's size changed with the silence).
    Worker(env.cfg, transcriber=env.fake).run()
    assert len(env.fake.calls) == 3 and env.md.read_text() == md
    # A re-rendering from the sidecar (a speaker named, a second reading aligned) keeps the line.
    data = json.loads(env.sidecar.read_text())
    env.store.rewrite_transcript(env.store.read_state()[env.memo]["json"], data, env.cfg.threshold, env.cfg.run_min,
                                 env.cfg.pause_ms, env.cfg.marker_spec)
    assert env.md.read_text() == md
    # The recorder's file fetched again, with the sound in it: silenced before anything reads it, and not
    # sent to the engine, because nothing else about the file changed.
    env.audio.write_bytes(env.original_audio)
    Worker(env.cfg, transcriber=env.fake).run()
    assert opus.unsilent(env.audio, plan["start_ms"], plan["end_ms"]) == 0 and len(env.fake.calls) == 3
    # A transcription from scratch on request: the engine hears words in the range again (here: the same
    # tokens), and they are dropped before anything is written.
    Worker(env.cfg, transcriber=env.fake).run(only=[env.memo], redo=True)
    assert len(env.fake.calls) == 4 and files_with(env) == []
    assert env.md.read_text().split("\n").count(marker) == 1 and len(json.loads(env.sidecar.read_text())["words"]) == 60
    assert redact.verify(env.cfg, env.store, env.memo)["ok"]


def beside_a_second_stream(path: Path) -> int:
    """Rewrite the file the way a Plaud recorder writes it: after each page of sound a page of another stream
    (its first one named, the rest 23 zero bytes). Returns how many such pages were added."""
    import struct

    def page(flags: int, granule: int, seq: int, body: bytes) -> bytes:
        head = b"OggS" + struct.pack("<BBqIIIB", 0, flags, granule, 777, seq, 0, 1) + bytes([len(body)])
        return head[:22] + struct.pack("<I", opus._crc(head + body)) + head[26:] + body

    out, n = [], 0
    for p in opus._read_pages(path.read_bytes()):
        out.append(p.raw)
        out.append(page(2 if n == 0 else 0, max(p.granule, 0), n, b"PLAUD.AI" + b"\0" * 15 if n == 0 else b"\0" * 23))
        n += 1
    stamp = path.stat().st_mtime
    path.write_bytes(b"".join(out))
    os.utime(path, (stamp, stamp))
    return n


def test_a_recorders_file_with_a_second_stream_of_zeros(env):
    plain = opus.info(env.audio)
    added = beside_a_second_stream(env.audio)
    env.store.write_state({**env.store.read_state(), env.memo: {**env.store.read_state()[env.memo], "fp": redact._fingerprint(env.audio)}})
    foreign = [p.raw for p in opus._read_pages(env.audio.read_bytes()) if p.foreign]
    info = opus.info(env.audio)
    assert len(foreign) == added == info["other_stream_pages"] and info["duration_ms"] == plain["duration_ms"] and info["packets"] == plain["packets"]
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    assert plan["audio"]["supported"]
    done = apply(env, plan["start_ms"], plan["end_ms"])
    assert done["audio"]["silenced"] > 1000 and done["audio"]["duration_diff_ms"] < 50
    heard = done["audio"]["heard"]
    assert heard["decoded"] and heard["rms_in_range"] <= opus.QUIET_RMS and heard["rms_outside_after"] > 1000
    assert [p.raw for p in opus._read_pages(env.audio.read_bytes()) if p.foreign] == foreign  # not a byte of the other stream
    assert redact.verify(env.cfg, env.store, env.memo)["ok"] and files_with(env) == []
    assert apply(env, plan["start_ms"], plan["end_ms"])["already"]


def test_a_second_stream_that_holds_data_is_refused(env):
    beside_a_second_stream(env.audio)
    raw = bytearray(env.audio.read_bytes())
    at = raw.rindex(b"OggS", 0, raw.rindex(b"OggS"))  # the last but one page is the other stream's? find one of serial 777
    pages = opus._read_pages(bytes(raw))
    offset = 0
    for p in pages:
        if p.foreign and not p.flags & 2:
            body_at = offset + len(p.raw) - 1
            raw[body_at] = 7
            fixed = bytes(raw[offset:offset + len(p.raw)])
            raw[offset + 22:offset + 26] = __import__("struct").pack("<I", opus._crc(fixed[:22] + b"\0\0\0\0" + fixed[26:]))
            break
        offset += len(p.raw)
    assert at >= 0
    env.audio.write_bytes(bytes(raw))
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    assert not plan["audio"]["supported"] and "second stream" in plan["audio"]["problem"]
    with pytest.raises(redact.RedactError, match="nothing was changed"):
        apply(env, plan["start_ms"], plan["end_ms"])
    assert not env.store.redactions_path.exists()


@pytest.mark.parametrize("from_ms, to_ms, bounds, words", [
    (0, 4_000, (0, 4_000), 8),  # at the start
    (84_100, 10**7, (83_800, SECONDS * 1000), 2),  # to the end, the asked end past the recording
])
def test_a_range_at_the_start_and_at_the_end(env, from_ms, to_ms, bounds, words):
    plan = redact.plan(env.cfg, env.store, env.memo, from_ms, to_ms)
    assert (plan["start_ms"], plan["end_ms"]) == bounds and plan["words"] == words
    done = apply(env, *bounds)
    lines = [x for x in env.md.read_text().split("\n") if x.strip()]
    assert lines.count(done["marker"]) == 1 and len(json.loads(env.sidecar.read_text())["words"]) == 90 - words
    assert redact.verify(env.cfg, env.store, env.memo)["ok"] and abs(opus.info(env.audio)["duration_ms"] - SECONDS * 1000) < 50


def test_what_apply_refuses(env):
    with pytest.raises(redact.RedactError, match="not the bounds plan gave"):
        apply(env, FROM_MS, TO_MS)
    with pytest.raises(redact.RedactError, match="whole recording"):
        apply(env, 0, SECONDS * 1000)
    env.audio.write_bytes(b"RIFF" + b"\0" * 64)  # not a file that can be silenced
    plan = redact.plan(env.cfg, env.store, env.memo, FROM_MS, TO_MS)
    assert not plan["audio"]["supported"] and "Ogg Opus" in plan["audio"]["problem"]
    with pytest.raises(redact.RedactError, match="nothing was changed"):
        apply(env)
    assert SECRET in env.md.read_text() and not env.store.redactions_path.exists()


@pytest.mark.parametrize("broken", ["_silence_in_place", "rewrite_transcript", "write_state"])
def test_an_apply_cut_short_is_finished_by_the_same_command(env, monkeypatch, broken):
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    target = redact if broken == "_silence_in_place" else Store

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(target, broken, boom)
        with pytest.raises(KeyboardInterrupt):
            apply(env, plan["start_ms"], plan["end_ms"])
    assert env.store.redactions_for(env.memo)[0]["state"] == "begun"  # on record from the first write on
    assert redact.verify(env.cfg, env.store, env.memo)["ok"] is False
    done = apply(env, plan["start_ms"], plan["end_ms"])
    assert not done["already"] and files_with(env) == [] and redact.verify(env.cfg, env.store, env.memo)["ok"]
    assert not [p for top in (env.cfg.import_dir, env.md.parent, env.cfg.data_dir) for p in top.glob(".*tmp")]


def test_a_worker_run_after_a_cut_short_apply_leaves_the_sound_to_the_command(env, monkeypatch):
    plan = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)
    with monkeypatch.context() as m:
        m.setattr(redact, "_silence_in_place", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            apply(env, plan["start_ms"], plan["end_ms"])
    loud = opus.unsilent(env.audio, plan["start_ms"], plan["end_ms"])
    Worker(env.cfg, transcriber=env.fake).run()  # a timed run comes first: it does not finish the job on its own
    assert opus.unsilent(env.audio, plan["start_ms"], plan["end_ms"]) == loud > 1000 and len(env.fake.calls) == 3
    again = redact.plan(env.cfg, env.store, env.memo, 30_000, 59_000)  # the bounds can be looked at once more
    assert (again["start_ms"], again["end_ms"]) == (plan["start_ms"], plan["end_ms"]) and len(again["edge"]["before"]) == 12
    apply(env, plan["start_ms"], plan["end_ms"])
    assert files_with(env) == [] and redact.verify(env.cfg, env.store, env.memo)["ok"]


def test_the_command_prints_one_object_and_no_content_in_errors(env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: env.cfg)

    def run(*args) -> tuple[int, dict]:
        capsys.readouterr()
        code = cli.main(["redact", *args, "--json"])
        out = capsys.readouterr().out.strip().split("\n")
        assert len(out) == 1
        return code, json.loads(out[0])

    code, plan = run("plan", "--memo", env.memo, "--from-ms", "30000", "--to-ms", "59000")
    assert code == 0 and (plan["start_ms"], plan["end_ms"]) == (24_800, 60_000)
    code, refused = run("apply", "--memo", env.memo, "--start-ms", "24800", "--end-ms", "60000", "--date", "2026-11-21")
    assert code == 2 and "--yes" in refused["error"] and SECRET in env.md.read_text()
    code, done = run("apply", "--memo", env.memo, "--start-ms", "24800", "--end-ms", "60000", "--date", "2026-11-21", "--yes")
    assert code == 0 and done["marker"] == "[poistettu 21.11.2026: 0:24–1:00]" and files_with(env) == []
    code, checked = run("verify", "--memo", env.memo)
    assert code == 0 and checked["ok"]
    code, listed = run("list")
    assert listed["memos"][env.memo] == [{"start_ms": 24_800, "end_ms": 60_000, "date": "2026-11-21", "state": "done"}]
    code, error = run("plan", "--memo", "ei-ole", "--from-ms", "0", "--to-ms", "1000")
    assert code == 2 and set(error) == {"error"}
    code, error = run("plan", "--memo", env.memo)
    assert code == 2 and "--from-ms" in error["error"]


def test_a_transcript_without_ranges_renders_as_before():
    words, _ = notation.tokens_to_words(TOKENS)
    speakers = notation.speaker_labels(words)
    plain = notation.render_blocks(words, 0.7, 4, 2500, speakers, 15)
    assert notation.layout(words, 0.7, 4, 2500, speakers, 15)["lines"] == plain == notation.render_blocks(words, 0.7, 4, 2500, speakers, 15, [])
    assert plain[0].startswith("**S1:** kivikatu") and plain[1].startswith("**S2:** latokatu") and plain[2].startswith("[00:20] ")
