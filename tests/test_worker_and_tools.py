import json
import time

import pytest

from tests.conftest import FakeTranscriber
from voicememos_mcp.config import load_config
from voicememos_mcp.tools import ToolError, Tools
from voicememos_mcp.worker import Worker


def run(cfg, transcriber, **kw):
    return Worker(cfg, transcriber=transcriber).run(**kw)


def test_first_run_transcribes_eligible_memos_only(cfg, fake):
    summary = run(cfg, fake)
    assert summary["ok"]
    titles = sorted(i["title"] for i in summary["transcribed"])
    assert titles == ["Car notes", "Morning walk"]  # not deleted, not excluded, not in iCloud
    assert len(fake.calls) == 2

    library = json.loads((cfg.data_dir / "library.json").read_text())
    assert library["excluded_count"] == 1
    assert all(m["title"] != "Secret plans" for m in library["memos"])
    status = {m["title"]: m["status"] for m in library["memos"]}
    assert status["Old memo in iCloud"] == "not_downloaded"
    assert status["Deleted one"] == "deleted"
    assert status["Morning walk"] == "transcribed"

    md = next(cfg.output_dir.rglob("*morning-walk_aaaa1111.md")).read_text()
    assert "Hello world (?)." in md and "## Markers" in md and "**Idea:** build a bird house" in md
    data = json.loads(next(cfg.output_dir.rglob("*morning-walk_aaaa1111.json")).read_text())
    assert data["schema"] == "voicememos-mcp/transcript@1"
    assert data["source_tokens"][1]["text"] == " wor"
    assert data["words"][1]["basis"] == "asr_confidence"

    worker = json.loads((cfg.data_dir / "worker.json").read_text())
    assert worker["transcribed_count"] == 2 and worker["library_readable"]


def test_imported_files_are_transcribed_like_memos(cfg, fake):
    import os

    folder = cfg.import_dir
    folder.mkdir(parents=True)
    audio = folder / "GMT20261031-102304_Recording.m4a"
    audio.write_bytes(b"\0" * 64)
    old = time.time() - 3600
    os.utime(audio, (old, old))
    (folder / (audio.name + ".json")).write_text(json.dumps(
        {"title": "Alhovuori meeting", "recorded": "2026-10-31T12:23:04+02:00", "duration_s": 120}))

    summary = run(cfg, fake)
    assert sorted(i["title"] for i in summary["transcribed"]) == ["Alhovuori meeting", "Car notes", "Morning walk"]
    library = json.loads((cfg.data_dir / "library.json").read_text())
    item = next(m for m in library["memos"] if m["title"] == "Alhovuori meeting")
    assert item["status"] == "transcribed" and item["folder"] == "Imported" and item["path"] == str(audio)
    assert library["import_dir"] == str(folder)
    md = next(cfg.output_dir.rglob("*alhovuori-meeting_*.md"))
    assert md.name.startswith("2026-10-31_1223_") and "Hello world (?)." in md.read_text()

    run(cfg, fake)
    assert len(fake.calls) == 3  # not transcribed again

    audio.unlink()  # the audio removed to save space: the transcript stays listed and readable
    run(cfg, fake)
    library = json.loads((cfg.data_dir / "library.json").read_text())
    item = next(m for m in library["memos"] if m["title"] == "Alhovuori meeting")
    assert item["status"] == "audio_removed" and item["recorded"] == "2026-10-31T12:23:04+02:00"
    assert item["folder"] == "Imported" and not item["local"]
    tools = Tools(cfg)
    assert "Hello world (?)." in tools.read(item["short_id"])["text"]
    assert [m["title"] for m in tools.list_memos(status="audio_removed")["memos"]] == ["Alhovuori meeting"]
    assert tools.search("bird house", since="2026-10-31", until="2026-10-31")["count"] == 1
    md.unlink(), md.with_suffix(".json").unlink()  # the transcript deleted as well: nothing left to list
    state = json.loads((cfg.data_dir / "state.json").read_text())
    state[item["id"]].pop("memo")
    (cfg.data_dir / "state.json").write_text(json.dumps(state))
    run(cfg, fake)
    library = json.loads((cfg.data_dir / "library.json").read_text())
    assert all(m["title"] != "Alhovuori meeting" for m in library["memos"])


def test_imports_are_not_pushed_to_roam(cfg, fake, monkeypatch):
    cfg.raw.update({"VOICEMEMOS_ROAM_ENABLED": "1", "ROAM_GRAPH": "example-graph"})
    monkeypatch.setenv("ROAM_API_TOKEN", "test-token")
    folder = cfg.import_dir
    folder.mkdir(parents=True)
    audio = folder / "2026-10-31_1223 Meeting.m4a"
    audio.write_bytes(b"\0" * 64)
    import os

    old = time.time() - 3600
    os.utime(audio, (old, old))
    sent = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json["append-data"][0]["string"])

            class R:
                ok = True
                status_code = 200
                text = ""
            return R()

    summary = Worker(cfg, transcriber=fake, roam_session=Session()).run()
    assert any("Meeting" in i["title"] for i in summary["transcribed"])
    assert len(sent) == 2 and not any("Meeting" in s for s in sent)


def test_import_command_copies_file_and_sidecar(cfg, tmp_path, monkeypatch, capsys):
    from voicememos_mcp import cli

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "probe_duration", lambda p: 42.0)
    src = tmp_path / "GMT20261031-102304_Recording.m4a"
    src.write_bytes(b"\0" * 16)
    assert cli.main(["import", str(src), "--title", "Alhovuori meeting"]) == 0
    assert (cfg.import_dir / src.name).exists() and src.exists()
    meta = json.loads((cfg.import_dir / (src.name + ".json")).read_text())
    assert meta == {"title": "Alhovuori meeting", "duration_s": 42.0}
    out = capsys.readouterr().out
    assert "Alhovuori meeting" in out and "0:42" in out and "next run" in out
    assert cli.main(["import", str(src)]) == 2  # already imported
    assert cli.main(["import", str(tmp_path / "notes.txt")]) == 2
    assert cli.main(["import", str(src), "--replace", "--recorded", "yesterday"]) == 2


def test_second_run_skips_done_memos(cfg, fake):
    run(cfg, fake)
    run(cfg, fake)
    assert len(fake.calls) == 2


def test_fresh_file_waits_for_icloud(cfg, fake, library):
    path = library / "20261014 090000-BBBB2222.qta"
    now = time.time()
    import os

    os.utime(path, (now, now))
    summary = run(cfg, fake)
    assert [i["title"] for i in summary["transcribed"]] == ["Morning walk"]


def test_since_limits_automatic_transcription(tmp_path, library, fake):
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "d"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "o"), "VOICEMEMOS_SINCE": "2999-01-01"}
    c = load_config(path=tmp_path / "none", env=env)
    assert run(c, fake)["transcribed"] == []


def test_failures_are_retried_after_growing_pauses(cfg):
    """Three quick failures used to end a memo for good; now each failure only lengthens the pause."""
    failing = FakeTranscriber(fail_for=("AAAA1111",))
    tries = lambda: len([c for c in failing.calls if "AAAA1111" in c])  # noqa: E731
    walk = lambda: next(m for m in json.loads((cfg.data_dir / "library.json").read_text())["memos"]  # noqa: E731
                        if m["title"] == "Morning walk")
    start = time.time()
    at = lambda seconds: Worker(cfg, transcriber=failing, clock=lambda: start + seconds).run()  # noqa: E731
    at(0), at(60), at(120)
    assert tries() == 1 and walk()["status"] == "pending"  # the minute runs in between wait
    at(400)
    assert tries() == 2
    at(1000), at(3000)
    assert tries() == 2  # an hour after the second failure
    worker = at(400 + 3700)
    assert tries() == 3 and walk()["status"] == "failed" and "simulated" in walk()["error"]
    alert = json.loads((cfg.data_dir / "worker.json").read_text())["alerts"][0]
    assert alert["kind"] == "memo_failed" and "aaaa1111" in alert["line"] and "Morning walk" not in alert["line"]
    assert worker["alerts"] == [alert]
    at(400 + 3700 + 3600)
    assert tries() == 3  # six hours now
    at(400 + 3700 + 7 * 3600)
    assert tries() == 4
    at(400 + 3700 + 20 * 3600)
    assert tries() == 4  # then once a day
    again = at(400 + 3700 + 32 * 3600)
    assert tries() == 5 and again["alerts"][0]["since"] == alert["since"]  # the alert keeps its first time

    fine = FakeTranscriber()
    summary = Worker(cfg, transcriber=fine, clock=lambda: start + 10 * 86400).run()
    assert len(fine.calls) == 1 and walk()["status"] == "transcribed" and summary["alerts"] == []


def test_queue_request_from_tools_is_processed(cfg, fake):
    run(cfg, FakeTranscriber())  # creates library.json
    tools = Tools(cfg)
    job = tools.transcribe(memos=["aaaa1111"], redo=True)
    assert job["status"] == "queued"
    with pytest.raises(ToolError):
        tools.transcribe()
    run(cfg, fake)
    status = tools.transcribe_status(job["job_id"])
    assert status["status"] == "done" and status["done"] == 1 and status["items"][0]["status"] == "transcribed"
    assert fake.calls == ["20261012 080000-AAAA1111.m4a"]
    assert tools.transcribe_status()["jobs"][0]["job_id"] == job["job_id"]


def test_queue_never_touches_excluded_memos(cfg, fake):
    run(cfg, FakeTranscriber())
    cfg.data_dir.joinpath("queue").mkdir(exist_ok=True)
    tools = Tools(cfg)
    job = tools.transcribe(since="2000-01-01", redo=True)
    run(cfg, fake)
    names = " ".join(fake.calls)
    assert "EEEE5555" not in names and "DDDD4444" not in names
    job_status = tools.transcribe_status(job["job_id"])
    assert job_status["total"] == 3  # includes the memo still in iCloud
    assert sorted(i["status"] for i in job_status["items"]) == ["not_downloaded", "transcribed", "transcribed"]


def test_tools_list_read_search_markers(cfg, fake):
    run(cfg, fake)
    tools = Tools(cfg)
    listed = tools.list_memos()
    assert listed["total"] == 3  # excluded and deleted memos are hidden
    assert listed["memos"][0]["title"] == "Car notes"
    assert tools.list_memos(status="not_downloaded")["memos"][0]["id"] == "cccc3333"

    page = tools.read("aaaa1111")
    assert "Hello world (?)." in page["text"] and page["next_offset"] is None
    assert "(?)" not in tools.read("aaaa1111", format="text", threshold=0.3)["text"]
    data = tools.read("aaaa1111", format="json")["data"]
    assert "source_tokens" not in data and data["words"][1]["flag"] == "uncertain"
    assert tools.read("cccc3333")["text"] is None

    hits = tools.search("bird")
    assert hits["count"] == 2 and "bird house" in hits["hits"][0]["snippets"][0]
    assert tools.search("walk")["hits"][0]["title_match"]
    assert tools.search("Secret")["count"] == 0
    with pytest.raises(ToolError):
        tools.search("(", regex=True)

    markers = tools.markers(label="idea")
    assert markers["total"] == 2 and markers["markers"][0]["text"] == "build a bird house"

    with pytest.raises(ToolError):
        tools.read("ee")  # too short
    with pytest.raises(ToolError):
        tools.read("eeee5555")  # excluded memo is unknown to the tools
    with pytest.raises(ToolError):
        tools.list_memos(since="14.10.2026")


def test_status_reports_worker_problems(cfg, fake, tmp_path):
    tools = Tools(cfg)
    assert "never run" in tools.status()["warnings"][0]
    broken = load_config(path=tmp_path / "none", env={**cfg.raw, "VOICEMEMOS_RECORDINGS_DIR": str(tmp_path / "x")})
    summary = Worker(broken, transcriber=fake).run()
    assert not summary["ok"] and not summary["library_readable"]
    assert "cannot read" in Tools(broken).status()["warnings"][0].lower()


def test_transcript_path_traversal_is_rejected(cfg, fake):
    run(cfg, fake)
    assert Tools(cfg).store.load_transcript("../data/state.json") is None


def test_roam_push_only_for_new_auto_transcripts(cfg, fake, monkeypatch):
    cfg.raw.update({"VOICEMEMOS_ROAM_ENABLED": "1", "ROAM_GRAPH": "example-graph"})
    monkeypatch.setenv("ROAM_API_TOKEN", "test-token")
    sent = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append((url, json, headers))

            class R:
                ok = True
                status_code = 200
                text = ""
            return R()

    Worker(cfg, transcriber=fake, roam_session=Session()).run()
    assert len(sent) == 2
    url, body, headers = sent[0]
    assert "example-graph" in url and headers["X-Authorization"] == "Bearer test-token"
    assert body["append-data"][0]["string"].startswith("🎙️ Voice memo")
    Worker(cfg, transcriber=fake, roam_session=Session()).run()
    assert len(sent) == 2  # no duplicates


def test_dry_run_saves_nothing(cfg, fake):
    summary = run(cfg, fake, dry_run=True)
    assert summary["transcribed"][0]["status"] == "dry_run"
    assert not (cfg.data_dir / "library.json").exists()
    assert not list(cfg.output_dir.rglob("*.md"))


def test_worker_marks_itself_running_and_publishes_progress(cfg, monkeypatch):
    seen = []

    class Spy(FakeTranscriber):
        def __call__(self, audio):
            seen.append(json.loads((cfg.data_dir / "worker.json").read_text()).get("running"))
            return super().__call__(audio)

    run(cfg, Spy())
    assert seen and all(seen)
    final = json.loads((cfg.data_dir / "worker.json").read_text())
    assert "running" not in final and final["transcribed_count"] == 2


def test_skip_keyword_in_title_prevents_transcription(cfg, fake, library, monkeypatch):
    import sqlite3

    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZENCRYPTEDTITLE = 'Car notes notranscript' WHERE ZUNIQUEID LIKE 'BBBB2222%'")
    db.commit()
    db.close()

    summary = run(cfg, fake)
    assert [i["title"] for i in summary["transcribed"]] == ["Morning walk"]
    lib = json.loads((cfg.data_dir / "library.json").read_text())
    car = next(m for m in lib["memos"] if m["title"] == "Car notes notranscript")
    assert car["status"] == "not_transcribed" and "notranscript" in car["error"]

    tools = Tools(cfg)
    assert tools.list_memos(title="car")["total"] == 1  # still visible, unlike an exclusion
    job = tools.transcribe(memos=["bbbb2222"])
    run(cfg, fake)
    status = tools.transcribe_status(job["job_id"])
    assert status["items"][0]["status"] == "skipped" and "rename" in status["items"][0]["error"]
    assert all("BBBB2222" not in c for c in fake.calls)

    # Renaming the memo makes it pending again.
    db = sqlite3.connect(library / "CloudRecordings.db")
    db.execute("UPDATE ZCLOUDRECORDING SET ZENCRYPTEDTITLE = 'Car notes' WHERE ZUNIQUEID LIKE 'BBBB2222%'")
    db.commit()
    db.close()
    summary = run(cfg, fake)
    assert [i["title"] for i in summary["transcribed"]] == ["Car notes"]
