import shutil
import subprocess
import sys


def test_syntax():
    subprocess.run(["bash", "-n", "whisper-npu"], check=True)


def test_help_exits_zero():
    out = subprocess.run(
        ["bash", "whisper-npu", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "USAGE" in out.stdout


def test_no_files_errors():
    out = subprocess.run(
        ["bash", "whisper-npu"],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "USAGE" in out.stderr


def test_requires_flm_binary():
    assert shutil.which("flm"), "flm (FastFlowLM) must be installed to run whisper-npu"


def test_dictate_syntax():
    subprocess.run(["bash", "-n", "npu-dictate"], check=True)


def test_dictate_help_exits_zero():
    out = subprocess.run(
        ["bash", "npu-dictate", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "USAGE" in out.stdout
    assert "push-to-talk" in out.stdout


def test_dictate_no_command_errors():
    out = subprocess.run(
        ["bash", "npu-dictate"],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "USAGE" in out.stderr


def test_dictate_rejects_unknown_command():
    out = subprocess.run(
        ["bash", "npu-dictate", "frobnicate"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 2
    assert "unknown command" in out.stderr


def test_dictate_status_is_readable():
    out = subprocess.run(
        ["bash", "npu-dictate", "status"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() in {"idle", "recording", "transcribing"}


def test_dictate_output_backend_present():
    assert shutil.which("wtype") or shutil.which("wl-copy"), (
        "npu-dictate needs wtype (to type) or wl-copy (clipboard fallback)"
    )


def test_dictate_recorder_present():
    assert shutil.which("pw-record"), "npu-dictate records with pw-record (pipewire)"


def test_chat_compiles():
    subprocess.run(
        [sys.executable, "-m", "py_compile", "npu-chat"], check=True
    )


def test_chat_help_exits_zero():
    out = subprocess.run(
        [sys.executable, "npu-chat", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--model" in out.stdout
    assert "Enter" in out.stdout


def test_chat_rejects_unknown_flag():
    out = subprocess.run(
        [sys.executable, "npu-chat", "--nope"],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0


def test_chat_recorder_and_ffmpeg_present():
    assert shutil.which("pw-record"), "npu-chat records with pw-record (pipewire)"
    assert shutil.which("ffmpeg"), "npu-chat normalises captures with ffmpeg"
