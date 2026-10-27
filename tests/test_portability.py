"""Cross-platform behaviour that only shows up on Windows in practice: console encoding, memory-mapped cache
files that cannot be deleted while mapped, video-writer fallbacks, and disk-space checks."""

import io
import os
import shutil
import sys

import pytest

from uapka import cli, video
from uapka.synth import open_writer
from uapka.video import load_frames, probe


def test_cli_prints_safely_on_a_cp1252_console(monkeypatch, lossless_clip, capsys):
    path, _ = lossless_clip
    # a redirected/legacy Windows console: cp1252 with strict errors raises on any non-ASCII character
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    with pytest.raises(UnicodeEncodeError):
        print("1\u03c3 \u00b1 \u2265 \u26a0")              # proves the stream is the hostile one
    assert cli.main(["info", path]) == 0
    print("after: 1\u03c3 \u00b1 \u2265 \u26a0", file=sys.stdout)   # must not raise once main() ran
    print("stderr: \u03c9", file=sys.stderr)
    sys.stdout.flush()
    text = sys.stdout.buffer.getvalue().decode("utf-8")
    assert "SHA-256:" in text and "after: 1\u03c3 \u00b1 \u2265 \u26a0" in text
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8" and sys.stdout.errors == "replace"


def test_windowed_build_without_stdio_gets_a_sink(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    cli.configure_stdio()
    print("hello \u03c9")
    print("hello", file=sys.stderr)
    assert sys.stdout is not None and sys.stderr is not None


def test_memmap_cache_survives_live_views_and_is_removed(lossless_clip, tmp_path):
    path, _ = lossless_clip
    fc = load_frames(probe(path), ram_budget_bytes=1, cache_dir=str(tmp_path))
    cache_file = fc._memmap_path
    assert cache_file and os.path.isfile(cache_file) and os.path.dirname(cache_file) == str(tmp_path)
    assert fc.frames.base is not None                    # frames is a view of the memmap
    held = fc[3]                                         # a caller keeps a frame (as the GUI does)
    fc.close()                                           # must not raise even while `held` maps the file
    assert fc._memmap_path is None and len(fc) == 0
    if os.name != "nt":
        assert not os.path.exists(cache_file)
    else:                                                # Windows: deferred until the view is gone
        assert not os.path.exists(cache_file) or cache_file in video._PENDING_REMOVAL
    del held
    video._remove_pending()
    assert not os.path.exists(cache_file)
    assert cache_file not in video._PENDING_REMOVAL


def test_memmap_refuses_when_the_cache_disk_is_full(monkeypatch, lossless_clip, tmp_path):
    path, _ = lossless_clip
    real = shutil.disk_usage
    monkeypatch.setattr(video.shutil, "disk_usage", lambda d: real(d)._replace(free=0))
    with pytest.raises(IOError, match="free"):
        load_frames(probe(path), ram_budget_bytes=1, cache_dir=str(tmp_path))
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".u8")]


def test_ram_budget_heuristics(monkeypatch):
    b = video.default_ram_budget_bytes()
    assert 256 * 1024 ** 2 <= b <= 2 * 1024 ** 3
    monkeypatch.setenv("UAPKA_RAM_BUDGET_MB", "300")
    assert video.default_ram_budget_bytes() == 300 * 1024 ** 2
    monkeypatch.setenv("UAPKA_CACHE_DIR", str(os.path.join("some", "dir")))
    assert video.default_cache_dir() == os.path.abspath(os.path.join("some", "dir"))


def test_stale_cache_sweep(tmp_path):
    old = tmp_path / "uapka_old.u8"
    old.write_bytes(b"\0" * 8)
    os.utime(old, (0, 0))
    fresh = tmp_path / "uapka_new.u8"
    fresh.write_bytes(b"\0" * 8)
    other = tmp_path / "keep.u8"
    other.write_bytes(b"\0")
    os.utime(other, (0, 0))
    assert video.sweep_stale_caches(str(tmp_path)) == 1
    assert not old.exists() and fresh.exists() and other.exists()


def test_synth_writer_falls_back_with_a_note(tmp_path):
    w, codec, note = open_writer(str(tmp_path / "x.avi"), "ZZZZ", 30.0, (64, 48))
    w.release()
    assert codec in ("FFV1", "HFYU") and "ZZZZ" in note and "lossless" in note
    w, codec, note = open_writer(str(tmp_path / "y.avi"), "FFV1", 30.0, (64, 48))
    w.release()
    assert codec == "FFV1" and note == ""
    with pytest.raises(IOError, match="No video writer"):
        open_writer(str(tmp_path / "z.nosuchcontainer"), "FFV1", 30.0, (64, 48))


def test_ffprobe_is_called_without_a_console_window(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"], calls["kw"] = cmd, kw

        class R:
            stdout = '{"streams": [{"codec_name": "ffv1"}]}'
        return R()
    monkeypatch.setattr(video.shutil, "which", lambda name: "C:\\ff\\ffprobe.exe")
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    assert video.run_ffprobe("clip.avi") == {"codec_name": "ffv1"}
    assert calls["cmd"][0] == "C:\\ff\\ffprobe.exe" and calls["cmd"][-1] == "clip.avi"
    assert "shell" not in calls["kw"] and calls["kw"]["creationflags"] == (
        getattr(video.subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)


def test_probe_reports_missing_file_clearly(tmp_path):
    with pytest.raises(IOError, match="No such file"):
        probe(str(tmp_path / "missing.mp4"))


def test_a_bare_path_argument_opens_the_gui(monkeypatch, lossless_clip):
    path, _ = lossless_clip
    opened = {}
    monkeypatch.setattr(cli, "cmd_gui", lambda a: opened.setdefault("path", a.path) or 0)
    assert cli.main([path]) == 0 and opened["path"] == path     # what drag-and-drop onto UAPKA.exe passes
    with pytest.raises(SystemExit):
        cli.main(["no-such-file-or-command"])


def test_check_env_runs_and_reports(capsys):
    from uapka import envcheck
    assert envcheck.main() == 0
    out = capsys.readouterr().out
    assert "FFmpeg backend" in out and "platform" in out and "Summary:" in out and out.isascii()
