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


def test_failures_are_counted_and_capped(cfg):
    failing = FakeTranscriber(fail_for=("AAAA1111",))
    for _ in range(4):
        run(cfg, failing)
    attempts = [c for c in failing.calls if "AAAA1111" in c]
    assert len(attempts) == 3  # VOICEMEMOS_MAX_ATTEMPTS
    library = json.loads((cfg.data_dir / "library.json").read_text())
    walk = next(m for m in library["memos"] if m["title"] == "Morning walk")
    assert walk["status"] == "failed" and "simulated" in walk["error"]


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
