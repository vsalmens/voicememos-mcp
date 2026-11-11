import datetime as dt
import json
import os
import stat
import time

import pytest

from tests.conftest import FakeTranscriber
from voicememos_mcp import plaud
from voicememos_mcp.config import load_config
from voicememos_mcp.worker import Worker

BASE = "https://plaud.test/api"
MP3 = b"ID3" + b"\0" * 200


class Response:
    def __init__(self, status=200, data=None, body=b"", headers=None):
        self.status_code, self._data, self._body, self.headers = status, data, body, headers or {}
        self.ok = status < 400
        self.text = json.dumps(data) if data is not None else ""

    def json(self):
        return self._data

    def iter_content(self, chunk_size=1):
        for k in range(0, len(self._body), 7):
            yield self._body[k:k + 7]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePlaudCloud:
    """The API's side: recordings by id, a login that expires, links that carry the audio."""

    def __init__(self, files):
        self.files = {f["id"]: f for f in files}
        self.audio = {f["id"]: MP3 for f in files}
        self.valid = {"access-1"}
        self.refreshes = 0
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url))
        assert url == BASE + plaud.REFRESH_PATH
        if data["refresh_token"] != "refresh-1":
            return Response(401, {"detail": "bad"})
        self.refreshes += 1
        token = f"access-{self.refreshes + 1}"
        self.valid = {token}
        return Response(200, {"access_token": token, "expires_in": 3600})

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append(("GET", url))
        if url.startswith("https://audio.test/"):
            assert not headers  # the presigned link gets no login header
            file_id = url.rsplit("/", 1)[1].split(".")[0]
            return Response(200, body=self.audio[file_id], headers={"Content-Type": "audio/mpeg"})
        if headers["Authorization"].removeprefix("Bearer ") not in self.valid:
            return Response(401, {"detail": "expired"})
        path = url.removeprefix(BASE)
        if path == "/open/third-party/files/":
            rows = sorted(self.files.values(), key=lambda f: f["id"])
            start = (params["page"] - 1) * params["page_size"]
            listed = [{k: v for k, v in f.items() if k != "presigned_url"} for f in rows[start:start + params["page_size"]]]
            return Response(200, {"data": listed, "page": params["page"]})
        file_id = path.rsplit("/", 1)[1]
        return Response(200, self.files[file_id]) if file_id in self.files else Response(404, {})


def recording(file_id, name, start="2026-11-09T08:15:00", minutes=30, url=True):
    return {"id": file_id, "name": name, "created_at": "2026-11-09T12:00:00", "start_at": start,
            "duration": minutes * 60_000, "presigned_url": f"https://audio.test/{file_id}.mp3" if url else None}


@pytest.fixture
def tokens(tmp_path):
    path = tmp_path / "plaud" / "tokens.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"access_token": "access-1", "refresh_token": "refresh-1", "token_type": "Bearer",
                                "expires_at": int((time.time() + 3600) * 1000)}))
    return path


def client(cloud, tokens, **kw):
    return plaud.Plaud(tokens, BASE, session=cloud, **kw)


def test_times_from_the_api_are_utc_unless_they_say_otherwise():
    utc = dt.timezone.utc
    assert plaud.parse_time("2026-11-09T08:15:00").astimezone(utc).isoformat() == "2026-11-09T08:15:00+00:00"
    assert plaud.parse_time("2026-11-09 08:15:00Z").astimezone(utc).hour == 8
    assert plaud.parse_time("2026-11-09T10:15:00+02:00").astimezone(utc).hour == 8
    assert plaud.parse_time(1794212100000) == plaud.parse_time(1794212100) == plaud.parse_time("1794212100000")
    assert plaud.parse_time("") is None and plaud.parse_time("soon") is None and plaud.parse_time(None) is None


def test_extension_from_link_content_type_or_first_bytes():
    assert plaud.extension_for("https://x/a/b.MP3?sig=1", "", b"") == ".mp3"
    assert plaud.extension_for("https://x/a/b.opus?sig=1", "", b"") == ".ogg"
    assert plaud.extension_for("https://x/a/b", "audio/x-m4a; q=1", b"") == ".m4a"
    assert plaud.extension_for("https://x/a/b", "application/octet-stream", b"OggS\0\0") == ".ogg"
    assert plaud.extension_for("https://x/a/b", "", b"\0\0\0\x20ftypM4A ") == ".m4a"
    assert plaud.extension_for("https://x/a/b.bin", "", b"\xff\xfb\x90\x00") == ".mp3"
    assert plaud.extension_for("https://x/a/b", "", b"hello") is None


def test_fetch_saves_audio_and_sidecar_once(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "Tiimipalaveri"), recording("b2", "11-09 Kävely", start=None, minutes=5),
                            recording("c3", "Still on the device", url=False)])
    imports, state = tmp_path / "imports", tmp_path / "plaud.json"

    result = plaud.fetch_new(client(cloud, tokens), imports, state)
    assert result["listed"] == 3 and sorted(result["fetched"]) == ["a1", "b2"]
    assert result["waiting"] == ["c3"] and result["failed"] == []
    audio = imports / "plaud_a1.mp3"
    assert audio.read_bytes() == MP3 and not list(imports.glob(".*"))
    meta = json.loads((imports / "plaud_a1.mp3.json").read_text())
    assert meta["title"] == "Tiimipalaveri" and meta["folder"] == "Plaud" and meta["duration_s"] == 1800
    when = dt.datetime.fromisoformat(meta["recorded"])
    assert when.astimezone(dt.timezone.utc).isoformat() == "2026-11-09T08:15:00+00:00"
    assert int(audio.stat().st_mtime) == int(when.timestamp())  # old enough for the worker at once
    # without start_at the upload time stands in
    assert json.loads((imports / "plaud_b2.mp3.json").read_text())["recorded"].startswith("2026-11-09T")

    cloud.files["c3"]["presigned_url"] = "https://audio.test/c3.mp3"  # uploaded since
    audio.unlink()  # removed by the user: not fetched again
    result = plaud.fetch_new(client(cloud, tokens), imports, state)
    assert result["fetched"] == ["c3"] and not audio.exists()
    assert sorted(json.loads(state.read_text())["files"]) == ["a1", "b2", "c3"]


def test_rename_in_plaud_updates_the_sidecar(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "2026-11-09 10:15:00")])
    imports, state = tmp_path / "imports", tmp_path / "plaud.json"
    plaud.fetch_new(client(cloud, tokens), imports, state)
    cloud.files["a1"]["name"] = "Budjettikokous"
    result = plaud.fetch_new(client(cloud, tokens), imports, state)
    assert result["fetched"] == []
    assert json.loads((imports / "plaud_a1.mp3.json").read_text())["title"] == "Budjettikokous"


def test_since_dry_run_and_limit(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "Old", start="2026-10-01T08:00:00"), recording("b2", "New 1"),
                            recording("c3", "New 2", start="2026-11-10T08:00:00")])
    imports, state = tmp_path / "imports", tmp_path / "plaud.json"
    result = plaud.fetch_new(client(cloud, tokens), imports, state, since="2026-11-01", dry_run=True)
    assert sorted(result["fetched"]) == ["b2", "c3"] and not imports.exists() and not state.exists()
    result = plaud.fetch_new(client(cloud, tokens), imports, state, since="2026-11-01", limit=1)
    assert result["fetched"] == ["b2"]  # oldest first
    with pytest.raises(plaud.PlaudError):
        plaud.fetch_new(client(cloud, tokens), imports, state, since="last week")


def test_expired_login_is_refreshed_and_written_back(tmp_path, tokens):
    stored = json.loads(tokens.read_text()) | {"expires_at": int((time.time() - 10) * 1000)}
    tokens.write_text(json.dumps(stored))
    cloud = FakePlaudCloud([recording("a1", "One")])
    cloud.valid = set()
    result = plaud.fetch_new(client(cloud, tokens), tmp_path / "imports", tmp_path / "plaud.json")
    assert result["fetched"] == ["a1"] and cloud.refreshes == 1
    saved = json.loads(tokens.read_text())
    assert saved["access_token"] == "access-2" and saved["refresh_token"] == "refresh-1"
    assert saved["expires_at"] > time.time() * 1000 and stat.S_IMODE(tokens.stat().st_mode) == 0o600


def test_rejected_token_is_refreshed_once_then_login_is_asked_for(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "One")])
    cloud.valid = set()  # revoked although not expired by the clock
    assert plaud.fetch_new(client(cloud, tokens), tmp_path / "i", tmp_path / "p.json")["fetched"] == ["a1"]

    tokens.write_text(json.dumps({"access_token": "x", "refresh_token": "gone"}))
    with pytest.raises(plaud.PlaudAuthError, match="plaud login"):
        plaud.fetch_new(client(cloud, tokens), tmp_path / "i", tmp_path / "p.json")
    tokens.unlink()
    with pytest.raises(plaud.PlaudAuthError, match="plaud login"):
        plaud.fetch_new(client(cloud, tokens), tmp_path / "i", tmp_path / "p.json")


def test_one_failed_download_does_not_stop_the_rest(tmp_path, tokens):
    cloud = FakePlaudCloud([recording("a1", "Empty"), recording("b2", "Fine")])
    cloud.audio["a1"] = b""
    cloud.files["b2"]["presigned_url"] = "https://audio.test/b2.mp3"
    result = plaud.fetch_new(client(cloud, tokens), tmp_path / "imports", tmp_path / "plaud.json")
    assert result["fetched"] == ["b2"] and result["failed"][0]["id"] == "a1"
    assert [p.name for p in (tmp_path / "imports").iterdir() if p.suffix != ".json"] == ["plaud_b2.mp3"]
    cloud.audio["a1"] = MP3  # tried again, but not on every run: after a pause that grows
    state = tmp_path / "plaud.json"
    calls = len(cloud.calls)
    again = plaud.fetch_new(client(cloud, tokens), tmp_path / "imports", state)
    assert again["fetched"] == [] and again["failing"] == 1 and len(cloud.calls) - calls <= 2  # only the listing
    failure = json.loads(state.read_text())["failures"]["a1"]
    assert failure["count"] == 1 and "empty" in failure["error"]
    later = plaud.fetch_new(client(cloud, tokens), tmp_path / "imports", state, clock=lambda: time.time() + 400)
    assert later["fetched"] == ["a1"] and later["failing"] == 0
    assert json.loads(state.read_text())["failures"] == {}


def test_worker_fetches_and_transcribes_plaud_recordings(tmp_path, library, tokens):
    env = {"VOICEMEMOS_RECORDINGS_DIR": str(library), "VOICEMEMOS_DATA_DIR": str(tmp_path / "data"),
           "VOICEMEMOS_OUTPUT_DIR": str(tmp_path / "out"), "VOICEMEMOS_WORKER": "none",
           "VOICEMEMOS_SINCE": "2026-11-01", "VOICEMEMOS_PLAUD": "1", "VOICEMEMOS_ROAM_ENABLED": "1",
           "ROAM_GRAPH": "g", "ROAM_API_TOKEN": "t"}
    cfg = load_config(path=tmp_path / "missing.env", env=env)
    cloud = FakePlaudCloud([recording("a1", "Tiimipalaveri")])
    fake = FakeTranscriber()

    class NoRoam:  # a Plaud recording is not a note of the day
        def post(self, *a, **k):
            raise AssertionError("Roam must not be called for an imported recording")

    worker = Worker(cfg, transcriber=fake, plaud_client=client(cloud, tokens), roam_session=NoRoam())
    summary = worker.run()
    assert summary["plaud"] == {"ok": True, "listed": 1, "fetched": 1, "waiting": 0, "failed": []}
    titles = [i["title"] for i in summary["transcribed"]]
    assert "Tiimipalaveri" in titles
    assert not list(cfg.output_dir.rglob("*tiimipalaveri_*"))  # apart from the dictated memos
    path = next(cfg.plaud_output_dir.rglob("*tiimipalaveri_*.md"))
    assert path.parent == cfg.data_dir / "plaud-transcripts" / "2026" and path.with_suffix(".json").exists()
    md = path.read_text()
    assert '\nfolder: "Plaud"\n' in md.split("---")[1]
    from voicememos_mcp.tools import Tools
    tools = Tools(cfg)
    entry = worker.state[next(k for k, v in worker.state.items() if str(v.get("markdown", "")).endswith(path.name))]
    assert entry["markdown"] == str(path) and tools.store.load_transcript(entry["json"])["memo"]["folder"] == "Plaud"
    assert tools.store.load_transcript("/etc/hosts") is None and tools.store.load_transcript("../state.json") is None
    worker.store.rewrite_transcript(entry["json"], tools.store.load_transcript(entry["json"]), 0.7, 4, 2500, cfg.marker_spec)
    memo_id = next(k for k, v in worker.state.items() if v is entry)
    assert "Hello" in tools.read(memo_id)["text"] and path.exists()  # the MCP tools read it where it is
    item = next(m for m in json.loads((cfg.data_dir / "library.json").read_text())["memos"] if m["title"] == "Tiimipalaveri")
    assert item["folder"] == "Plaud" and item["status"] == "transcribed"

    calls = len(fake.calls)
    assert worker.run()["plaud"]["fetched"] == 0 and len(fake.calls) == calls

    # a transcript written among the dictated memos (0.7.0) moves to the Plaud folder on the next run
    old = cfg.output_dir / "2026" / path.name
    path.rename(old), path.with_suffix(".json").rename(old.with_suffix(".json"))
    state = json.loads((cfg.data_dir / "state.json").read_text())
    state[memo_id].update(markdown=f"2026/{old.name}", json=f"2026/{old.with_suffix('.json').name}")
    (cfg.data_dir / "state.json").write_text(json.dumps(state))
    worker.run()
    assert path.exists() and path.with_suffix(".json").exists() and not old.exists() and len(fake.calls) == calls
    assert json.loads((cfg.data_dir / "state.json").read_text())[memo_id]["markdown"] == str(path)
    voice = [v["markdown"] for k, v in worker.state.items() if k != memo_id and v.get("ok")]
    assert voice and all(not m.startswith("/") for m in voice)  # dictated memos stay where they were


def test_plaud_failure_is_a_warning_and_voice_memos_still_run(cfg, tmp_path):
    cfg.raw["VOICEMEMOS_PLAUD"] = "1"
    cfg.raw["VOICEMEMOS_PLAUD_TOKENS"] = str(tmp_path / "nowhere" / "tokens.json")
    summary = Worker(cfg, transcriber=FakeTranscriber()).run()
    assert summary["ok"] and summary["plaud"]["ok"] is False and "plaud login" in summary["warnings"][0]
    assert len(summary["transcribed"]) == 2
    assert json.loads((cfg.data_dir / "worker.json").read_text())["warnings"]


def test_plaud_is_off_by_default(cfg):
    summary = Worker(cfg, transcriber=FakeTranscriber()).run()
    assert "plaud" not in summary and not (cfg.data_dir / "plaud.json").exists()
