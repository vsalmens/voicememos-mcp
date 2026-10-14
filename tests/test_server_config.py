import asyncio

from voicememos_mcp.cli import launchd_plist
from voicememos_mcp.config import load_config, parse_env_file
from voicememos_mcp.server import build_server
from voicememos_mcp.tools import Tools


def test_server_exposes_tools(cfg):
    server = build_server(Tools(cfg))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"voicememos_status", "voicememos_list", "voicememos_read", "voicememos_search",
                     "voicememos_markers", "voicememos_transcribe", "voicememos_transcribe_status",
                     "voicememos_second_read", "voicememos_compare_readers"}


def test_config_file_and_environment(tmp_path):
    path = tmp_path / "config.env"
    path.write_text('# comment\nVOICEMEMOS_LANGUAGES="fi, en"\nexport VOICEMEMOS_MARKERS=idea=Idea,note\n'
                    "VOICEMEMOS_CONF_THRESHOLD='0.6'\n", encoding="utf-8")
    assert parse_env_file(path)["VOICEMEMOS_LANGUAGES"] == "fi, en"
    cfg = load_config(path=path, env={"VOICEMEMOS_CONF_THRESHOLD": "0.8", "HOME_UNRELATED": "x"})
    assert cfg.languages == ["fi", "en"]
    assert cfg.markers == {"idea": "Idea", "note": "Note"}
    assert cfg.threshold == 0.8  # environment wins
    assert "HOME_UNRELATED" not in cfg.raw


def test_launchd_plist_watches_recordings_and_queue(cfg):
    plist = launchd_plist(cfg, ["/x/python", "-m", "voicememos_mcp", "sync"])
    assert "<string>local.voicememos-mcp.sync</string>" in plist
    assert str(cfg.data_dir / "queue") in plist and str(cfg.recordings_dir) in plist
