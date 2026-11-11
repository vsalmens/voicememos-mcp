"""What goes wrong around the transcription itself: a changed file, a broken state file, a killed
run, an engine that cannot be reached, and what a run with nothing to do leaves alone."""
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import FakeTranscriber
from tests.test_plaud import FakePlaudCloud, client, recording, tokens  # noqa: F401 - fixture
from voicememos_mcp import cli, plaud, segment
from voicememos_mcp.config import load_config
from voicememos_mcp.library import read_imports
from voicememos_mcp.soniox import Soniox, SonioxError, SonioxUnavailable
from voicememos_mcp.store import StateError, Store, write_json_atomic
from voicememos_mcp.tools import Tools
from voicememos_mcp.worker import Busy, Worker, single_instance

HAS_FFMPEG = bool(segment.find_tool("ffmpeg") and segment.find_tool("ffprobe"))
WALK = "20261012 080000-AAAA1111.m4a"
WALK_ID = "AAAA1111-0000-0000-0000-000000000001"
REPO = Path(__file__).resolve().parent.parent


def run(cfg, transcriber, **kw):
    return Worker(cfg, transcriber=transcriber, **kw).run()


def load(path):
    return json.loads(Path(path).read_text())


def memo(cfg, title):
    return next(m for m in load(cfg.data_dir / "library.json")["memos"] if m["title"] == title)


def aged(path, seconds=3600):
    old = time.time() - seconds
    os.utime(path, (old, old))


# ----------------------------------------------------------------------------- a changed file

def test_a_changed_recording_is_transcribed_again(cfg, fake, library):
    run(cfg, fake)
    assert len(fake.calls) == 2
    aged(library / WALK, 7200)  # iCloud downloading the same audio again: a new time, the same file
    run(cfg, fake)
    assert len(fake.calls) == 2
    (library / WALK).write_bytes(b"\0" * 5000)  # recorded on, or edited
    aged(library / WALK)
    summary = run(cfg, fake)
    assert [i["title"] for i in summary["transcribed"]] == ["Morning walk"] and len(fake.calls) == 3
    run(cfg, fake)
    assert len(fake.calls) == 3


def test_the_earlier_transcript_stands_until_the_new_one_is_written(cfg, fake, library):
    run(cfg, fake)
    (library / WALK).write_bytes(b"\0" * 5000)
    aged(library / WALK)
    run(cfg, FakeTranscriber(fail_for=("AAAA1111",)))
    walk = memo(cfg, "Morning walk")
    assert walk["status"] == "pending" and "simulated" in walk["error"] and walk["transcript"]["markdown"]
    assert "Hello" in Tools(cfg).read("aaaa1111")["text"]


def test_import_replace_is_transcribed_again(cfg, fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "probe_duration", lambda p: 42.0)
    src = tmp_path / "meeting.m4a"
    src.write_bytes(b"\0" * 10)
    aged(src)
    assert cli.main(["import", str(src)]) == 0
    run(cfg, fake)
    assert "meeting.m4a" in fake.calls
    src.write_bytes(b"\0" * 999)
    aged(src)
    assert cli.main(["import", str(src), "--replace"]) == 0
    run(cfg, fake)
    assert fake.calls.count("meeting.m4a") == 2


# ----------------------------------------------------------------------------- failures

class Offline(FakeTranscriber):
    def __call__(self, audio):
        self.calls.append(audio.name)
        raise SonioxUnavailable("Soniox cannot be reached: no route to host")


def test_an_engine_that_cannot_be_reached_is_not_the_memos_failure(cfg):
    offline = Offline()
    start = time.time()
    runs = [Worker(cfg, transcriber=offline, clock=lambda: start + 60 * minute).run() for minute in range(30)]
    # not counted, the other memo not even tried in the same run, and not asked every minute
    assert len(runs[0]["failed"]) == 1 and "not available" in runs[0]["warnings"][0]
    assert all(len(r["failed"]) <= 1 for r in runs) and runs[-1]["failed"] == []
    assert 4 <= len(offline.calls) <= 12
    state = load(cfg.data_dir / "state.json")
    assert all(entry.get("attempts", 0) == 0 for entry in state.values())
    assert memo(cfg, "Morning walk")["status"] == "pending"
    assert load(cfg.data_dir / "worker.json")["alerts"] == []
    fake = FakeTranscriber()
    Worker(cfg, transcriber=fake, clock=lambda: start + 3600).run()
    assert len(fake.calls) == 2  # back online: everything is transcribed


def test_a_new_file_starts_from_zero_tries(cfg, library):
    failing = FakeTranscriber(fail_for=("AAAA1111",))
    start = time.time()
    for later in (0, 400, 4100):
        Worker(cfg, transcriber=failing, clock=lambda: start + later).run()
    assert memo(cfg, "Morning walk")["status"] == "failed"
    (library / WALK).write_bytes(b"\0" * 777)
    aged(library / WALK)
    fake = FakeTranscriber()
    Worker(cfg, transcriber=fake, clock=lambda: start + 4200).run()  # no day's wait for another file
    assert fake.calls == [WALK] and memo(cfg, "Morning walk")["status"] == "transcribed"


def test_a_request_that_cannot_be_run_does_not_come_back(cfg, fake, monkeypatch):
    run(cfg, fake)
    tools = Tools(cfg)
    bad = tools.second_read(memos=["aaaa1111"], realign=True)["job_id"]
    good = tools.transcribe(memos=["bbbb2222"], redo=True)["job_id"]
    monkeypatch.setattr(Worker, "realign_memo", lambda self, memo: 1 / 0)
    summary = run(cfg, fake)
    assert summary["ok"] and [j["job_id"] for j in summary["jobs"]] == [bad, good]
    assert tools.transcribe_status(bad)["status"] == "failed" and "division" in tools.transcribe_status(bad)["error"]
    assert tools.transcribe_status(good)["status"] == "done"
    assert not list((cfg.data_dir / "queue").glob("*.json"))


# ----------------------------------------------------------------------------- the state file

def test_an_unreadable_state_file_stops_the_run(cfg, fake):
    run(cfg, fake)
    state = cfg.data_dir / "state.json"
    good = state.read_text()
    state.write_text("")  # what a power cut could leave behind
    with pytest.raises(StateError):
        Store(cfg.data_dir, cfg.output_dir).read_state()
    summary = run(cfg, fake)
    assert not summary["ok"] and summary["state_readable"] is False and len(fake.calls) == 2
    worker = load(cfg.data_dir / "worker.json")
    assert worker["alerts"][0]["kind"] == "state_unreadable" and state.read_text() == ""
    status = Tools(cfg).status()
    assert status["alerts"][0]["kind"] == "state_unreadable" and "state.json.bak" in status["warnings"][0]
    since = worker["alerts"][0]["since"]
    run(cfg, fake)
    assert load(cfg.data_dir / "worker.json")["alerts"][0]["since"] == since

    state.write_text(good)
    assert run(cfg, fake)["alerts"] == [] and len(fake.calls) == 2


def test_state_keeps_a_backup_of_the_file_before_the_last_change(cfg, fake, library):
    run(cfg, FakeTranscriber(fail_for=("AAAA1111",)))
    first = (cfg.data_dir / "state.json").read_text()
    Worker(cfg, transcriber=fake, clock=lambda: time.time() + 400).run()
    assert (cfg.data_dir / "state.json.bak").read_text() == first
    assert load(cfg.data_dir / "state.json")[WALK_ID]["ok"]
    assert not list(cfg.data_dir.glob(".state.json*"))


def test_atomic_write_leaves_no_temporary_file_when_it_fails(tmp_path):
    with pytest.raises(TypeError):
        write_json_atomic(tmp_path / "x.json", {"a": object()})
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------- a killed run

def test_a_run_killed_after_the_engine_answered_does_not_pay_twice(cfg, fake, monkeypatch):
    finish = Worker._finish

    def killed(self, *args, **kwargs):
        raise KeyboardInterrupt  # the process stops during the second reading or the write

    monkeypatch.setattr(Worker, "_finish", killed)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, fake)
    assert len(fake.calls) == 1 and list((cfg.data_dir / "tmp").glob("*/tokens.json"))
    monkeypatch.setattr(Worker, "_finish", finish)
    summary = run(cfg, fake)
    assert len(summary["transcribed"]) == 2 and len(fake.calls) == 2  # one memo each, none twice
    assert not list((cfg.data_dir / "tmp").glob("*/tokens.json"))
    usage = Store(cfg.data_dir, cfg.output_dir).recent_usage()
    assert next(iter(usage.values()))["soniox_files"] == 2


class FlakySoniox:
    """Soniox's side, with a connection that drops while the transcription is polled."""

    def __init__(self, drops=0, die_at=None):
        self.headers, self.drops, self.die_at = {}, drops, die_at
        self.polls, self.deleted, self.uploads = 0, [], 0

    def request(self, method, url, **kw):
        class R:
            ok, status_code, text = True, 200, ""

            def __init__(self, data):
                self._d = data

            def json(self):
                return self._d

        if url.endswith("/v1/files"):
            self.uploads += 1
            return R({"id": "f1"})
        if url.endswith("/v1/transcriptions"):
            return R({"id": "t1"})
        if url.endswith("/transcript"):
            return R({"tokens": [{"text": "Hi"}]})
        self.polls += 1
        if self.die_at == self.polls:
            raise SystemExit(143)
        if self.polls <= self.drops:
            raise ConnectionError("connection reset")
        return R({"status": "completed" if self.polls > self.drops + 1 else "transcribing"})

    def delete(self, url, **kw):
        class R:
            ok, status_code = True, 200
        self.deleted.append(url.split(".test")[1])
        return R()


def test_a_dropped_connection_while_polling_is_waited_out(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"0")
    session = FlakySoniox(drops=9)  # more than the retries of one request
    pauses = []
    engine = Soniox("k", "https://soniox.test", session=session, poll_interval=0, sleep=pauses.append)
    assert engine.transcribe(audio, ["en"]) == [{"text": "Hi"}]
    assert session.uploads == 1 and session.polls == 11 and max(pauses) <= 60

    # ... but not for longer than a transcription may take
    engine = Soniox("k", "https://soniox.test", session=FlakySoniox(drops=10 ** 6), poll_interval=0,
                    poll_timeout=-1, sleep=lambda s: None)
    with pytest.raises(SonioxUnavailable):
        engine.transcribe(audio, ["en"])


def test_what_a_killed_run_left_at_soniox_is_deleted_by_the_next(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"0")
    pending = tmp_path / "soniox-pending.json"

    class Killed(FlakySoniox):
        def delete(self, url, **kw):
            raise SystemExit(137)  # no time for the clean-up either

    with pytest.raises(SystemExit):
        Soniox("k", "https://soniox.test", session=Killed(die_at=1), poll_interval=0,
               pending_path=pending).transcribe(audio, ["en"])
    assert load(pending) == ["/v1/files/f1", "/v1/transcriptions/t1"]
    session = FlakySoniox()
    assert Soniox("k", "https://soniox.test", session=session, pending_path=pending).cleanup_leftovers() == 2
    assert sorted(session.deleted) == ["/v1/files/f1", "/v1/transcriptions/t1"] and not pending.exists()

    # an ordinary run leaves nothing to clean
    Soniox("k", "https://soniox.test", session=FlakySoniox(), poll_interval=0, pending_path=pending,
           sleep=lambda s: None).transcribe(audio, ["en"])
    assert not pending.exists()


def test_a_refused_key_is_not_a_failure_of_the_recording(tmp_path):
    class Refusing(FlakySoniox):
        def request(self, method, url, **kw):
            class R:
                ok, status_code, text = False, 401, "bad key"
            return R()

    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"0")
    with pytest.raises(SonioxUnavailable):
        Soniox("k", "https://soniox.test", session=Refusing()).transcribe(audio, ["en"])

    class Rejecting(FlakySoniox):
        def request(self, method, url, **kw):
            class R:
                ok, status_code, text = False, 400, "unsupported audio"
            return R()

    with pytest.raises(SonioxError) as caught:
        Soniox("k", "https://soniox.test", session=Rejecting()).transcribe(audio, ["en"])
    assert not isinstance(caught.value, SonioxUnavailable)


# ----------------------------------------------------------------------------- long recordings that are not AAC

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are needed")
def test_a_long_mp3_is_cut_into_mp3_parts(tmp_path, library):
    """A Plaud recorder writes MP3; parts copied into an .m4a container are refused by ffmpeg."""
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_WORKER": "none",
           "VOICEMEMOS_MAX_DURATION_MIN": "0.3", "VOICEMEMOS_SEGMENT_MARGIN_MIN": "0.05",
           "VOICEMEMOS_SEGMENT_SILENCE_WINDOW_MIN": "0", "VOICEMEMOS_SINCE": "2999-01-01"}
    cfg = load_config(path=tmp_path / "missing.env", env=env)
    cfg.import_dir.mkdir(parents=True)
    audio = cfg.import_dir / "plaud_x1.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=23",
                    "-c:a", "libmp3lame", str(audio)], check=True)
    aged(audio)
    (cfg.import_dir / "plaud_x1.mp3.json").write_text(json.dumps(
        {"title": "Long meeting", "recorded": "2999-02-01T10:00:00+02:00", "duration_s": 23.0, "folder": "Plaud"}))
    fake = FakeTranscriber()
    summary = Worker(cfg, transcriber=fake).run()
    assert fake.calls == ["part1.mp3", "part2.mp3"]
    assert summary["transcribed"][0]["segments"] == 2 and memo(cfg, "Long meeting")["status"] == "transcribed"
    usage = next(iter(Store(cfg.data_dir, cfg.output_dir).recent_usage().values()))
    assert usage["soniox_files"] == 2 and usage["soniox_audio_min"] == pytest.approx(23 / 60, abs=0.02)


# ----------------------------------------------------------------------------- Plaud

def test_plaud_pagination_goes_on_until_an_empty_page(tmp_path, tokens):
    class Capped(FakePlaudCloud):  # a server that gives at most 2 per page, whatever is asked
        def get(self, url, params=None, **kw):
            return super().get(url, params={**params, "page_size": 2} if params else None, **kw)

    cloud = Capped([recording(f"r{k}", f"Recording {k}") for k in range(5)])
    result = plaud.fetch_new(client(cloud, tokens), tmp_path / "imports", tmp_path / "plaud.json")
    assert result["listed"] == 5 and len(result["fetched"]) == 5

    class Stuck(FakePlaudCloud):  # ... and one that ignores the page number does not loop for ever
        def get(self, url, params=None, **kw):
            return super().get(url, params={**params, "page": 1} if params else None, **kw)

    cloud = Stuck([recording("a1", "One")])
    assert plaud.fetch_new(client(cloud, tokens), tmp_path / "i2", tmp_path / "p2.json")["listed"] == 1
    assert len([c for c in cloud.calls if c[1].endswith("/files/")]) == 2


def test_plaud_backs_off_after_429(tmp_path, tokens):
    class Busy429(FakePlaudCloud):
        limited = True

        def get(self, url, params=None, headers=None, **kw):
            if self.limited and "third-party" in url:
                self.calls.append(("GET", url))
                from tests.test_plaud import Response
                return Response(429, {"detail": "slow down"}, headers={"Retry-After": "1200"})
            return super().get(url, params=params, headers=headers, **kw)

    cloud = Busy429([recording("a1", "One")])
    imports, state = tmp_path / "imports", tmp_path / "plaud.json"
    start = time.time()
    first = plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start)
    assert first["rate_limited"] and first["fetched"] == []
    calls = len(cloud.calls)
    for minute in range(1, 20):  # the worker's next runs do not ask at all
        assert plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start + 60 * minute)["rate_limited"]
    assert len(cloud.calls) == calls
    cloud.limited = False
    done = plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start + 1260)
    assert done["fetched"] == ["a1"] and "rate_limited" not in load(state)


def test_plaud_idle_call_is_small_and_writes_nothing(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "One", start="2026-11-09T08:15:00"),
                            recording("b2", "Two", start="2026-11-10T08:15:00")])
    seen = []
    listing = cloud.get

    def get(url, params=None, **kw):
        if params:
            seen.append(params.get("start_date"))
        return listing(url, params=params, **kw)

    cloud.get = get
    imports, state = tmp_path / "imports", tmp_path / "plaud.json"
    start = time.time()
    plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start)
    assert seen[0] is None  # the first listing is of everything
    written = state.stat().st_mtime_ns, state.read_text()
    del seen[:]
    plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start + 60)
    newest = plaud.parse_time("2026-11-10T08:15:00").timestamp()
    assert seen[0] == int((newest - plaud.RECENT_DAYS * 86400) * 1000)  # only the last days
    assert (state.stat().st_mtime_ns, state.read_text()) == written
    del seen[:]
    plaud.fetch_new(client(cloud, tokens), imports, state, clock=lambda: start + 3700)
    assert seen[0] is None  # and everything again once an hour


def test_plaud_command_does_not_run_beside_the_worker(cfg, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    cfg.data_dir.mkdir(parents=True)
    with single_instance(cfg.data_dir / "worker.lock"):
        assert cli.main(["plaud"]) == 1
    assert "worker is running" in capsys.readouterr().err
    with pytest.raises(Busy):
        with single_instance(cfg.data_dir / "worker.lock"), single_instance(cfg.data_dir / "worker.lock"):
            pass


def plaud_cfg(tmp_path, library, tokens):  # noqa: F811
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_WORKER": "none",
           "VOICEMEMOS_SINCE": "2026-11-01", "VOICEMEMOS_PLAUD": "1", "VOICEMEMOS_PLAUD_TOKENS": str(tokens)}
    return load_config(path=tmp_path / "missing.env", env=env)


def test_a_title_given_later_reaches_the_transcript_without_the_engine(tmp_path, library, tokens):  # noqa: F811
    cfg = plaud_cfg(tmp_path, library, tokens)
    cloud = FakePlaudCloud([recording("a1", "2026-11-09 10:15:00")])
    fake = FakeTranscriber()
    worker = Worker(cfg, transcriber=fake, plaud_client=client(cloud, tokens))
    worker.run()
    path = next(cfg.plaud_output_dir.rglob("*.md"))
    before, calls = path.read_text(), len(fake.calls)
    assert 'title: "2026-11-09 10:15:00"' in before and "plaud_a1.mp3" in fake.calls

    cloud.files["a1"]["name"] = 'Budjettikokous: "Q1"'  # Plaud's AI names it after its summary
    summary = worker.run()
    assert summary["retitled"] == 1 and summary["plaud"]["renamed"] == 1 and len(fake.calls) == calls
    assert [p.name for p in cfg.plaud_output_dir.rglob("*.md")] == [path.name]  # the path stays
    after = path.read_text()
    assert 'title: "Budjettikokous: \\"Q1\\""' in after and '\n# Budjettikokous: "Q1"\n' in after
    changed = [(a, b) for a, b in zip(before.split("\n"), after.split("\n")) if a != b]
    assert len(changed) == 2  # the front matter line and the heading, nothing else
    assert load(path.with_suffix(".json"))["memo"]["title"] == 'Budjettikokous: "Q1"'
    assert memo(cfg, 'Budjettikokous: "Q1"')["status"] == "transcribed"
    assert Tools(cfg).read(memo(cfg, 'Budjettikokous: "Q1"')["short_id"])["text"].startswith("---\ntitle: \"Budj")

    stamp = path.stat().st_mtime_ns
    assert "retitled" not in worker.run() and path.stat().st_mtime_ns == stamp and len(fake.calls) == calls


def test_a_transcript_from_before_0_8_learns_its_title_once(cfg, fake, library):
    run(cfg, fake)
    state = load(cfg.data_dir / "state.json")
    for entry in state.values():
        entry.pop("title", None)
    (cfg.data_dir / "state.json").write_text(json.dumps(state))
    import sqlite3
    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZENCRYPTEDTITLE='Evening walk' WHERE ZUNIQUEID=?", (WALK_ID,))
    db.commit()
    db.close()
    assert run(cfg, fake)["retitled"] == 1 and len(fake.calls) == 2
    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert 'title: "Evening walk"' in md and "# Evening walk" in md


def test_plaud_alerts(tmp_path, library, tokens):  # noqa: F811
    cfg = plaud_cfg(tmp_path, library, tokens)
    cloud = FakePlaudCloud([recording("a1", "One")])
    cloud.audio["a1"] = b""
    start = time.time()
    at = lambda s: Worker(cfg, transcriber=FakeTranscriber(), plaud_client=client(cloud, tokens),  # noqa: E731
                          clock=lambda: start + s).run()
    first = at(0)
    assert first["plaud"]["ok"] is False and first["alerts"] == []  # one failure is not an alert yet
    assert at(60)["plaud"]["failing"] == 1
    later = at(700)
    assert [a["kind"] for a in later["alerts"]] == ["plaud_fetch"] and "1 Plaud recording" in later["alerts"][0]["line"]
    assert Tools(cfg).status()["alerts"] == later["alerts"]
    cloud.audio["a1"] = b"ID3" + b"\0" * 50
    assert at(5000)["alerts"] == []

    tokens.unlink()  # the login is gone: that needs a person at once
    refused = Worker(cfg, transcriber=FakeTranscriber(), clock=lambda: start + 5100).run()
    assert [a["kind"] for a in refused["alerts"]] == ["plaud_login"] and "plaud login" in refused["alerts"][0]["line"]
    assert set(refused["alerts"][0]) == {"kind", "line", "since"}


# ----------------------------------------------------------------------------- a run with nothing to do

def test_an_idle_run_writes_only_worker_json(cfg, fake, capsys, monkeypatch, caplog):
    run(cfg, fake)
    files = [cfg.data_dir / name for name in ("library.json", "state.json")]
    before = {f: (f.stat().st_ino, f.stat().st_mtime_ns) for f in files}
    seen = []
    monkeypatch.setattr(Store, "write_worker", lambda self, data: seen.append(data))
    summary = run(cfg, fake)
    assert {f: (f.stat().st_ino, f.stat().st_mtime_ns) for f in files} == before
    assert len(seen) == 1 and "running" not in seen[0] and seen[0]["finished"] and seen[0]["alerts"] == []
    assert summary["ok"] and len(fake.calls) == 2


def test_an_idle_timed_run_prints_nothing(cfg, fake, capsys, monkeypatch):
    import voicememos_mcp.worker as worker_module

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(worker_module, "soniox_transcriber", lambda cfg, pending=None: fake)
    assert cli.main(["sync"]) == 0
    assert '"transcribed"' in capsys.readouterr().out  # a run that did something says so
    assert cli.main(["sync"]) == 0
    assert capsys.readouterr().out == ""
    assert cli.main(["sync", "--limit", "1"]) == 0  # asked for by a person
    assert '"ok": true' in capsys.readouterr().out


def test_logs_are_set_aside_when_they_grow(cfg):
    logs = cfg.data_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "worker.log").write_bytes(b"x" * 3000)
    (logs / "worker.err.log").write_bytes(b"x" * 100)
    cfg.raw["VOICEMEMOS_LOG_MAX_MB"] = "0.001"
    cli.rotate_logs(cfg)
    assert sorted(p.name for p in logs.iterdir()) == ["worker.err.log", "worker.log.1"]
    cfg.raw["VOICEMEMOS_LOG_MAX_MB"] = "0"
    (logs / "worker.log").write_bytes(b"x" * 3000)
    cli.rotate_logs(cfg)
    assert (logs / "worker.log").exists()


def test_a_dropped_file_is_probed_once(tmp_path):
    folder = tmp_path / "imports"
    folder.mkdir()
    audio = folder / "note.m4a"
    audio.write_bytes(b"\0" * 10)
    given = folder / "meeting.m4a"
    given.write_bytes(b"\0" * 10)
    (folder / "meeting.m4a.json").write_text(json.dumps({"title": "Meeting", "duration_s": 90}))
    probed = []

    def probe(path):
        probed.append(path.name)
        return 12.3456

    for _ in range(3):
        memos = {m.filename: m for m in read_imports(folder, probe)}
    assert probed == ["note.m4a"] and memos["note.m4a"].duration_s == 12.346 and memos["meeting.m4a"].duration_s == 90
    assert load(folder / "meeting.m4a.json") == {"title": "Meeting", "duration_s": 90}
    audio.write_bytes(b"\0" * 20)  # replaced by hand: probed again
    read_imports(folder, probe), read_imports(folder, probe)
    assert probed == ["note.m4a", "note.m4a"]
    del probed[:]
    for _ in range(3):  # a file ffprobe cannot read is not asked about every minute either
        assert read_imports(folder, lambda p: probed.append(1))[0].duration_s in (None, 90.0)
    assert probed == []


def test_a_too_long_memo_is_warned_about_once(tmp_path, library, caplog):
    import sqlite3

    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZDURATION=? WHERE ZUNIQUEID=?", (30040.8, WALK_ID))
    db.commit()
    db.close()
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_WORKER": "none",
           "VOICEMEMOS_SEGMENT_LONG": "0"}
    cfg = load_config(path=tmp_path / "missing.env", env=env)
    fake = FakeTranscriber()
    with caplog.at_level(logging.WARNING, logger="voicememos_mcp.worker"):
        run(cfg, fake)
        assert len([r for r in caplog.records if WALK_ID in r.getMessage()]) == 1
        caplog.clear()
        run(cfg, fake)
        assert not [r for r in caplog.records if WALK_ID in r.getMessage()]
    assert load(cfg.data_dir / "worker.json")["skipped"][0]["short_id"] == "aaaa1111"  # still reported


# ----------------------------------------------------------------------------- usage, check, install

def test_usage_is_counted_per_month_and_shown(cfg, fake, capsys, monkeypatch):
    def reader(audio):
        return {"text": "Hello world. Idea: build a bird house.", "model": "gemini-test",
                "usage": {"totalTokenCount": 1200, "promptTokenCount": 1000, "candidatesTokenCount": 200}}

    Worker(cfg, transcriber=fake, second_reader=reader).run()
    usage = Tools(cfg).status()["usage"]
    month = time.strftime("%Y-%m")
    assert usage[month]["soniox_files"] == 2 and usage[month]["soniox_audio_min"] == pytest.approx(20 / 60, abs=0.01)
    assert usage[month]["gemini_tokens"] == 2400 and usage[month]["gemini_calls"] == 2
    assert usage[month]["gemini_input_tokens"] == 2000 and usage[month]["gemini_output_tokens"] == 400

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cfg.__class__, "secret", lambda self, name: "k")
    cli.main(["check"])
    out = capsys.readouterr().out
    assert f"usage {month}:" in out and "2,400 tokens" in out
    assert "2 transcribed, 0 waiting, 0 changed" in out


def test_check_says_what_the_next_run_would_do(cfg, fake, library, capsys, monkeypatch):
    run(cfg, fake)
    (library / WALK).write_bytes(b"\0" * 4000)
    aged(library / WALK)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cfg.__class__, "secret", lambda self, name: "k")
    cli.main(["check"])
    assert "1 transcribed, 0 waiting, 1 changed since they were transcribed" in capsys.readouterr().out
    (cfg.data_dir / "state.json").write_text("{")
    assert cli.main(["check"]) == 1 and "NOT READABLE" in capsys.readouterr().out


def test_install_script_skips_the_secret_question_without_a_terminal(tmp_path):
    text = (REPO / "install.sh").read_text()
    function = text[text.index("store_secret() {"):text.index("store_secret SONIOX_API_KEY")]
    script = ("set -euo pipefail\nSERVICE=x\nsecurity() { return 1; }\n" + function
              + "store_secret SONIOX_API_KEY\nstore_secret GEMINI_API_KEY\necho reached-the-launchd-step\n")
    result = subprocess.run(["bash", "-c", script], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert result.returncode == 0 and "reached-the-launchd-step" in result.stdout
    assert "skipped" in result.stdout
    assert subprocess.run(["bash", "-n", str(REPO / "install.sh")]).returncode == 0
    assert '> "$PLIST"' not in text  # the plist is written beside and moved into place
