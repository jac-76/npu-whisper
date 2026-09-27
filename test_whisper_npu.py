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


def test_chat_no_transcript_flag_exists():
    out = subprocess.run(
        [sys.executable, "npu-chat", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--no-transcript" in out.stdout
    assert "--no-speak" in out.stdout
    assert "--voice" in out.stdout


def test_chat_cleans_up_on_sigterm():
    """atexit does not run on signals; a handler must remove the session dir."""
    import glob
    import signal as sig
    import time as t

    before = set(glob.glob("/tmp/npu-chat-*"))
    proc = subprocess.Popen(
        [sys.executable, "npu-chat", "--no-speak", "--no-transcript"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t.sleep(2)
    proc.send_signal(sig.SIGTERM)
    proc.wait(timeout=15)
    t.sleep(0.5)
    assert set(glob.glob("/tmp/npu-chat-*")) <= before


def test_chat_declares_trim_threshold():
    src = open("npu-chat").read()
    assert "NPU_CHAT_TRIM_DB" in src
    assert "silenceremove" in src


def test_chat_llm_url_is_separable():
    """ASR must stay on the NPU while the LLM can live elsewhere."""
    out = subprocess.run(
        [sys.executable, "npu-chat", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--llm-url" in out.stdout
    src = open("npu-chat").read()
    assert "LLM_URL" in src
    # ASR endpoint must not be repointed by --llm-url
    assert 'f"{URL}/v1/audio/transcriptions"' in src


def _load_chat():
    """Import npu-chat as a module despite having no .py extension."""
    import types

    mod = types.ModuleType("npu_chat")
    mod.__dict__["__name__"] = "npu_chat"
    exec(compile(open("npu-chat").read(), "npu-chat", "exec"), mod.__dict__)
    return mod


def test_for_speech_strips_markup_piper_would_pronounce():
    """piper says "asterisk-sterisk-seven" for **7**, so markup must go."""
    m = _load_chat()
    assert m.for_speech("You now have **7** apples.") == "You now have 7 apples."
    assert m.for_speech("Use `ls -la` to list.") == "Use ls -la to list."
    assert m.for_speech("# Heading\nbody") == "Heading\nbody"
    assert m.for_speech("- one\n- two") == "one\ntwo"
    assert m.for_speech("See [the docs](http://x.y) now.") == "See the docs now."
    assert m.for_speech("That is *very* good.") == "That is very good."


def test_for_speech_keeps_underscores_in_identifiers():
    m = _load_chat()
    assert m.for_speech("snake_case_name stays") == "snake_case_name stays"


def test_default_model_is_the_benchmarked_one():
    m = _load_chat()
    assert m.DEFAULT_LLM == "qwen3.5:4b"
