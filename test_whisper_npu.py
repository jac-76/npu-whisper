import json
import os
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


def test_self_facts_remote_llm_claims_no_hardware_it_cannot_see():
    """A remote LLM used to be described as llama.cpp on the laptop's RTX 4070."""
    m = _load_chat()
    s = m.self_facts("qwen3.5:9b", None, False, "ollama",
                     m.llm_where_default("http://192.168.1.36:11434"))
    assert "served by ollama on a separate machine at 192.168.1.36" in s
    assert "not on the NPU" in s
    assert "4070" not in s
    assert m.llm_where_default("http://127.0.0.1:8080") == "this laptop"
    told = m.self_facts("x", None, False, None, "the desktop's RTX 3060 Ti")
    assert "the desktop's RTX 3060 Ti" in told


def _stub_server(routes: dict):
    """Serve fixed JSON bodies on 127.0.0.1; returns (base_url, server)."""
    import http.server
    import json
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


def test_models_listed_from_remote_llm_server():
    m = _load_chat()
    base, srv = _stub_server({
        "/v1/models": {"data": [{"id": "qwen3:8b"}, {"id": "granite4.2:3b"}]},
        "/api/version": {"version": "0.33.3"},
    })
    try:
        m.LLM_URL = base
        assert m.installed_models() == ["granite4.2:3b", "qwen3:8b"]
        assert m.llm_server_kind(base) == "ollama"
    finally:
        srv.shutdown()


def test_llama_server_detected_and_dead_server_is_none():
    m = _load_chat()
    base, srv = _stub_server({"/props": {"default_generation_settings": {}}})
    try:
        assert m.llm_server_kind(base) == "llama.cpp"
    finally:
        srv.shutdown()
    assert m.remote_models("http://127.0.0.1:9") == []


def test_llm_url_defaults_from_env_and_dead_server_fails_fast():
    import os

    env = dict(os.environ, NPU_CHAT_LLM_URL="http://127.0.0.1:9")
    out = subprocess.run(
        [sys.executable, "npu-chat", "--no-speak", "--no-transcript"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env,
        timeout=30,
    )
    assert out.returncode == 1
    assert "http://127.0.0.1:9 is unreachable" in out.stdout
    assert "NPU_CHAT_LLM_URL" in out.stdout


def test_bench_llm_compiles():
    subprocess.run([sys.executable, "-m", "py_compile", "bench_llm.py"], check=True)


def test_thinking_disabled_only_for_ollama():
    """Qwen3.5 on ollama spent all 512 tokens thinking and returned nothing."""
    m = _load_chat()
    h = [{"role": "user", "content": "hi"}]
    assert m.request_body("qwen3.5:9b", h, "ollama")["reasoning_effort"] == "none"
    assert "reasoning_effort" not in m.request_body("qwen3.5:4b", h, None)
    assert m.request_body("qwen3.8-27b", h, "llama.cpp")["reasoning_effort"] == "none"


# --- voice -> coding agent (--harness claude) --------------------------------

FIXTURE = "tests/fixtures/claude-stream.jsonl"
FAKE = os.path.abspath("tests/fake_claude.py")


def _chat_in(tmp_path):
    """Load npu-chat with its session dir in tmp_path (no signal handlers)."""
    m = _load_chat()
    m._SESSION_DIR = str(tmp_path)
    return m


def _git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return str(path)


def test_parse_event_on_captured_claude_stream():
    """Fixture is real Claude Code 2.1.284 output, three turns in one process."""
    m = _load_chat()
    evs = [e for line in open(FIXTURE) for e in m.parse_event(line)]
    kinds = [e["kind"] for e in evs]
    assert kinds.count("result") == 3
    assert kinds.count("init") == 3
    assert {"kind": "tool", "name": "Read",
            "input": {"file_path": "/work/repo/tally/config.py"}} in evs
    results = [e for e in evs if e["kind"] == "result"]
    assert results[0]["text"] == "7341" and not results[0]["is_error"]
    assert results[2]["denials"][0]["tool_input"]["command"] == "cat /etc/hostname"
    assert any(e["kind"] == "tool_error" and e["text"].startswith("Permission to use Bash")
               for e in evs)
    # thinking blocks and the thinking_tokens flood are dropped
    assert all(e["kind"] != "text" or "thinking" not in e for e in evs)
    assert m.parse_event("not json") == [] and m.parse_event('{"type":"x"}') == []


def test_harness_multi_turn_in_one_process(tmp_path):
    m = _chat_in(tmp_path)
    log = tmp_path / "argv.jsonl"
    os.environ["FAKE_CLAUDE_LOG"] = str(log)
    try:
        h = m.ClaudeHarness(str(tmp_path), "qwen3.5:4b-64k", "http://h:11434",
                            m.allow_rules(str(tmp_path), []), binary=FAKE)
        r1 = [e for e in h.send("first") if e["kind"] == "result"][0]
        r2 = [e for e in h.send("second") if e["kind"] == "result"][0]
        assert (r1["text"], r2["text"]) == ("echo: first", "echo: second")
        assert h.session_id == "sess-1"
        runs = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(runs) == 1, "both turns must go to the same process"
        assert runs[0]["base_url"] == "http://h:11434"
        assert "--resume" not in runs[0]["argv"]
    finally:
        h.stop()
        del os.environ["FAKE_CLAUDE_LOG"]


def test_harness_resumes_after_stop(tmp_path):
    m = _chat_in(tmp_path)
    log = tmp_path / "argv.jsonl"
    os.environ["FAKE_CLAUDE_LOG"] = str(log)
    try:
        h = m.ClaudeHarness(str(tmp_path), "m", "http://h:11434", [], binary=FAKE)
        list(h.send("one"))
        h.stop()
        list(h.send("two"))
        runs = [json.loads(x)["argv"] for x in log.read_text().splitlines()]
        assert len(runs) == 2
        assert runs[1][runs[1].index("--resume") + 1] == "sess-1"
        h.reset()
        assert h.session_id is None
    finally:
        h.stop()
        del os.environ["FAKE_CLAUDE_LOG"]


def test_harness_death_and_timeout_raise(tmp_path):
    m = _chat_in(tmp_path)
    h = m.ClaudeHarness(str(tmp_path), "m", "http://h:11434", [], binary=FAKE)
    try:
        list(h.send("die"))
        raise AssertionError("expected HarnessError")
    except m.HarnessError as exc:
        assert "model load failed" in str(exc)
    assert not h.alive()
    try:
        list(h.send("hang", timeout=1))
        raise AssertionError("expected HarnessError")
    except m.HarnessError as exc:
        assert "no result after" in str(exc)
    assert not h.alive()


def test_harness_env_is_a_copy(tmp_path):
    m = _chat_in(tmp_path)
    before = {k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    h = m.ClaudeHarness(str(tmp_path), "qwen3.5:4b-64k", "http://h:11434", [])
    env = h.env()
    assert env["ANTHROPIC_BASE_URL"] == "http://h:11434"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.5:4b-64k"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    after = {k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    assert before == after
    argv = h.argv()
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert "--bare" in argv


def test_workdir_guard(tmp_path):
    import pytest

    m = _load_chat()
    plain = tmp_path / "plain"
    plain.mkdir()
    for bad in (str(plain), os.path.expanduser("~"), "/", str(tmp_path / "nope")):
        with pytest.raises(m.HarnessError):
            m.check_workdir(bad)
    repo = _git_repo(tmp_path / "repo")
    assert m.check_workdir(repo) == os.path.realpath(repo)


def test_allow_rules_are_scoped():
    """A bare Read rule let the model read /etc/hostname and ~/.bashrc."""
    m = _load_chat()
    rules = m.allow_rules("/w", ["Bash(.venv/bin/python -m pytest:*)",
                                 "Bash(./run-tests.sh)", "Bash(make test)"])
    assert "Read" not in rules and "Edit" not in rules
    assert "Read(./**)" in rules and "Edit(./**)" in rules
    assert "Bash(git diff)" in rules
    assert not any(r.startswith("Bash(git") and ":*" in r for r in rules)
    assert "Bash(/w/.venv/bin/python -m pytest:*)" in rules
    assert "Bash(/w/run-tests.sh)" in rules
    assert "Bash(/w/make test)" not in rules
    prompt = m.agent_prompt(rules)
    assert ".venv/bin/python -m pytest <args>" in prompt
    assert "/w/" not in prompt


def test_confirm_gate():
    """whisper invents sentences from room noise; the agent must not act on them."""
    m = _load_chat()
    assert m.confirm_transcript("fix it", ask=lambda _: "") == "fix it"
    assert m.confirm_transcript("Thank you.", ask=lambda _: "x") is None
    assert m.confirm_transcript("fix the NPU", ask=lambda _: "fix the NPU bug") == "fix the NPU bug"

    def eof(_):
        raise EOFError
    assert m.confirm_transcript("anything", ask=eof) is None


def test_confirm_action_ignores_keys_typed_before_the_question(tmp_path):
    """An Enter pressed to barge in, or during transcription, sat in the
    terminal buffer and answered the hardware question before it was shown."""
    import pty
    import time

    child = tmp_path / "child.py"
    child.write_text(
        "import time, types\n"
        "m = types.ModuleType('npu_chat')\n"
        f"exec(compile(open({os.path.abspath('npu-chat')!r}).read(), 'npu-chat', 'exec'), m.__dict__)\n"
        "time.sleep(0.5)\n"
        "print('RESULT', m.confirm_action('Set the duct fan to 5?'), flush=True)\n"
    )
    master, slave = pty.openpty()
    p = subprocess.Popen([sys.executable, "-u", str(child)], stdin=slave,
                         stdout=subprocess.PIPE, text=True)
    os.close(slave)
    try:
        time.sleep(0.1)
        os.write(master, b"y\n")  # the stale key, typed before the question
        for line in p.stdout:
            if "Set the duct fan to 5?" in line:
                break
        os.write(master, b"x\n")  # the real answer
        rest = p.stdout.read()
        p.wait(timeout=10)
    finally:
        os.close(master)
        if p.poll() is None:
            p.kill()
    assert "RESULT False" in rest


def test_confirm_action_answers():
    m = _load_chat()
    assert m.confirm_action("q?", ask=lambda _: "y") is True
    assert m.confirm_action("q?", ask=lambda _: "YES") is True
    assert m.confirm_action("q?", ask=lambda _: "x") is False
    # Enter alone cancels: it is also the push-to-talk and barge-in key.
    assert m.confirm_action("q?", ask=lambda _: "") is False

    def eof(_):
        raise EOFError
    assert m.confirm_action("q?", ask=eof) is False


def test_spoken_summary_and_api_error():
    m = _load_chat()
    s = m.spoken_summary("Fixed **it**. Tests pass. One. Two. Three.")
    assert s == "Fixed it. Tests pass. One. The rest is on screen."
    assert m.spoken_summary("Done.") == "Done."
    assert m.is_api_error("API Error: 500 llama-server process has terminated")
    assert not m.is_api_error("The API returns 500 on bad input.")


def test_describe_tool_is_workdir_relative():
    m = _load_chat()
    assert m.describe_tool("Read", {"file_path": "/w/tally/cli.py"}, "/w") == "Read tally/cli.py"
    assert m.describe_tool("Bash", {"command": "git status"}, "/w") == "Bash git status"


def test_harness_cli_refuses_bad_setups(tmp_path):
    def run(*a):
        return subprocess.run([sys.executable, "npu-chat", *a], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=30)
    r = run("--harness", "claude")
    assert r.returncode == 2 and "--workdir" in r.stderr
    r = run("--workdir", str(tmp_path))
    assert r.returncode == 2 and "only apply with --harness" in r.stderr
    r = run("--harness", "claude", "--workdir", str(tmp_path))
    assert r.returncode == 2 and "not inside a git repository" in r.stderr


# --- streamed speech and barge-in ---------------------------------------------

def test_sentence_splitter_boundaries():
    m = _load_chat()
    sp = m.SentenceSplitter()
    assert sp.feed("Hello there. How") == ["Hello there. "]
    assert sp.feed(" are you?") == []           # no whitespace after ? yet
    assert sp.flush() == ["How are you?"]
    sp = m.SentenceSplitter()
    assert sp.feed("Pi is 3.14 roughly and e.g") == []   # no split inside numbers
    assert sp.feed(" more.\nNext line") == ["Pi is 3.14 roughly and e.g more.\n"]
    assert sp.flush() == ["Next line"]


def test_sentence_splitter_drops_fenced_code():
    m = _load_chat()
    sp = m.SentenceSplitter()
    out = sp.feed("Run this:\n```py\nprint(1)\n") + sp.feed("```\nDone. ")
    assert out == ["Run this:\n", "Done. "]
    sp = m.SentenceSplitter()
    assert sp.feed("Code: ```x = 1") == ["Code: "]
    assert sp.flush() == []                      # unterminated fence is never spoken


def _speaker(m, tmp_path, delay):
    """Speaker with a fake synth (writes the text) and a fake player that
    logs what it 'plays' and sleeps for `delay` seconds."""
    log = tmp_path / "played.txt"

    def factory(_voice):
        def synth(text, wav):
            open(wav, "w").write(text)
            return True
        return synth

    player = [sys.executable, "-c",
              "import sys,time; open(sys.argv[1],'a').write(open(sys.argv[2]).read()+'\\n');"
              f" time.sleep({delay})", str(log)]
    return m.Speaker("voice.onnx", synth_factory=factory, player=player), log


def test_speaker_plays_in_order_and_times_first_audio(tmp_path):
    m = _chat_in(tmp_path)
    sp, log = _speaker(m, tmp_path, 0.05)
    sp.begin()
    for piece in ["One is **bold**. ", "Two. ", "Three"]:
        sp.feed(piece)
    sp.finish()
    sp.wait_idle(10)
    assert log.read_text().split("\n")[:3] == ["One is bold.", "Two.", "Three"]
    assert sp.first_audio is not None and sp.first_audio < 5


def test_speaker_stop_silences_and_drops_the_queue(tmp_path):
    """Barge-in: Enter while it talks must cut it off, not finish the reply."""
    import time as t

    m = _chat_in(tmp_path)
    sp, log = _speaker(m, tmp_path, 3.0)
    sp.say("First sentence. Second sentence. Third sentence. Fourth.")
    assert sp.wait_first_audio(5) is not None
    deadline = t.time() + 5
    while not log.exists() and t.time() < deadline:   # wait until it is audibly playing
        t.sleep(0.02)
    t0 = t.time()
    sp.stop()
    sp.wait_idle(10)
    assert t.time() - t0 < 2.0, "stop() must not wait for playback to finish"
    t.sleep(0.3)
    assert log.read_text().count("\n") == 1, "queued sentences must be dropped"
    sp.say("After barge-in.")
    sp.wait_idle(10)
    assert log.read_text().strip().split("\n")[-1] == "After barge-in."


FAKE_KOKORO = """
import json, sys
open(sys.argv[0] + ".starts", "a").write(sys.argv[-1] + "\\n")
print("ready", flush=True)
for line in sys.stdin:
    req = json.loads(line)
    open(req["wav"], "w").write(req["text"])
    print("ok", flush=True)
"""


def _kokoro_in(m, tmp_path, worker_src):
    """Point npu-chat at a fake Kokoro install: the venv's python is this
    Python, the voices file has two English voices and a Japanese one."""
    import zipfile

    d = tmp_path / "kokoro"
    (d / "venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, d / "venv" / "bin" / "python")
    (d / "kokoro-v1.0.onnx").write_bytes(b"model")
    with zipfile.ZipFile(d / "voices-v1.0.bin", "w") as z:
        for name in ("af_heart", "bm_george", "jf_alpha"):
            z.writestr(name + ".npy", b"")
    worker = tmp_path / "npu-kokoro"
    worker.write_text(worker_src)
    m.KOKORO_DIR, m.KOKORO_WORKER = str(d), str(worker)
    return d


def test_kokoro_voices_resolve_only_when_installed(tmp_path):
    m = _chat_in(tmp_path)
    d = _kokoro_in(m, tmp_path, FAKE_KOKORO)
    assert m.find_voice("kokoro:af_heart") == "kokoro:af_heart"
    assert m.find_voice("kokoro:af_nobody") is None
    assert "kokoro:bm_george" in m.available_voices()
    assert "kokoro:jf_alpha" not in m.available_voices()     # not English
    assert m.voice_label("kokoro:af_heart") == "kokoro:af_heart"
    assert m.voice_label("/v/en_GB-cori-high.onnx") == "en_GB-cori-high"
    (d / "kokoro-v1.0.onnx").unlink()
    assert m.find_voice("kokoro:af_heart") is None


def test_speaker_keeps_one_kokoro_process_for_every_sentence(tmp_path):
    """Kokoro loads in about 0.4 s, so it must not restart per sentence."""
    m = _chat_in(tmp_path)
    _kokoro_in(m, tmp_path, FAKE_KOKORO)
    log = tmp_path / "played.txt"
    player = [sys.executable, "-c",
              "import sys; open(sys.argv[1],'a').write(open(sys.argv[2]).read()+'\\n')",
              str(log)]
    sp = m.Speaker("kokoro:af_heart", player=player)
    sp.say("One. Two. Three.")
    sp.wait_idle(10)
    assert log.read_text().split("\n")[:3] == ["One.", "Two.", "Three."]
    assert (tmp_path / "npu-kokoro.starts").read_text() == "af_heart\n"
    sp.set_voice("kokoro:bm_george")
    sp.say("Four.")
    sp.wait_idle(10)
    assert log.read_text().split("\n")[3] == "Four."
    assert (tmp_path / "npu-kokoro.starts").read_text() == "af_heart\nbm_george\n"


def test_kokoro_that_will_not_start_mutes_instead_of_hanging(tmp_path, capsys):
    import time

    m = _chat_in(tmp_path)
    _kokoro_in(m, tmp_path, 'print("error: No module named kokoro_onnx", flush=True)\n')
    sp = m.Speaker("kokoro:af_heart", player=["true"])
    sp.say("Nobody hears this.")
    t0 = time.time()
    sp.wait_idle(10)
    assert time.time() - t0 < 5
    assert ("kokoro af_heart did not start (error: No module named kokoro_onnx)"
            in capsys.readouterr().out)


def test_kokoro_process_is_out_of_reach_of_ctrl_c(tmp_path):
    """Ctrl+C cancels a turn, and the terminal sends SIGINT to its whole
    foreground process group: npu-kokoro must not be in it, or the first
    Ctrl+C would leave the rest of the session silent."""
    m = _chat_in(tmp_path)
    _kokoro_in(m, tmp_path, "import os, sys\n"
               "open(sys.argv[0] + '.pgrp', 'w').write(str(os.getpgrp()))\n"
               "print('ready', flush=True)\n"
               "sys.stdin.read()\n")
    synth = m._kokoro_synth_factory("af_heart")
    try:
        assert int((tmp_path / "npu-kokoro.pgrp").read_text()) != os.getpgrp()
    finally:
        synth.close()


def test_self_facts_names_the_kokoro_voice():
    m = _load_chat()
    s = m.self_facts("x", "kokoro:af_heart")
    assert "your voice is kokoro af_heart, which runs on the CPU" in s
    assert "piper" not in s


def test_real_kokoro_writes_speech(tmp_path):
    """The real npu-kokoro against the installed model; skipped without it."""
    import pytest
    import wave

    m = _chat_in(tmp_path)
    if "af_heart" not in m.kokoro_voices():
        pytest.skip(f"Kokoro is not installed in {m.KOKORO_DIR}")
    synth = m._kokoro_synth_factory("af_heart")
    wav = str(tmp_path / "s.wav")
    try:
        assert synth("Hello there.", wav)
    finally:
        synth.close()
    with wave.open(wav) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() > 0.3 * 24000


def test_confirm_flag_exists_for_chat_mode():
    out = subprocess.run([sys.executable, "npu-chat", "--help"],
                         capture_output=True, text=True, check=True)
    assert "--confirm" in out.stdout


# --- speech gate (Silero VAD) ---------------------------------------------------

def _wav16k(path, samples):
    import array
    import wave as w

    with w.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(array.array("h", samples).tobytes())
    return str(path)


def test_vad_off_without_model(tmp_path):
    """No model -> no filtering (the pre-gate behaviour), never a crash."""
    m = _load_chat()
    m.VAD_MODEL = str(tmp_path / "missing.onnx")
    assert m.vad_peak(_wav16k(tmp_path / "s.wav", [0] * 16000)) is None


def test_vad_scores_silence_low_and_speech_high(tmp_path):
    import pytest

    m = _load_chat()
    if not os.path.exists(m.VAD_MODEL):
        pytest.skip("Silero VAD model not installed")
    silence = _wav16k(tmp_path / "silence.wav", [0] * 32000)
    if m.vad_peak(silence) is None:
        pytest.skip("numpy/onnxruntime not importable here")
    assert m.vad_peak(silence) < m.VAD_MIN
    voice = m.find_voice(m.DEFAULT_VOICE)
    if not (voice and shutil.which("piper-tts") and shutil.which("ffmpeg")):
        pytest.skip("piper voice not available to synthesise speech")
    raw, speech = tmp_path / "raw.wav", tmp_path / "speech.wav"
    subprocess.run(["piper-tts", "-m", voice, "-f", str(raw)],
                   input=b"Fix the bug in the stats module and run the tests.",
                   capture_output=True, check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
                    "-ar", "16000", "-ac", "1", str(speech)], check=True)
    assert m.vad_peak(str(speech)) > 0.9


def test_vad_threshold_sits_between_measured_noise_and_quiet_speech():
    """Measured on this mic: noise peaked <= 0.074, quiet speech >= 0.163."""
    m = _load_chat()
    assert 0.074 < m.VAD_MIN < 0.163


def test_prewarm_loads_on_ollama_only(tmp_path):
    """A cold model cost ~5 s on the first reply; prewarm loads it at startup."""
    import http.server
    import threading
    import time as t

    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            hits.append((self.path, json.loads(body)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        m = _load_chat()
        m.prewarm(base, "qwen3.5:9b-16k", "llama.cpp")
        m.prewarm(base, "qwen3.5:9b-16k", None)
        m.prewarm(base, "qwen3.5:9b-16k", "ollama")
        deadline = t.time() + 5
        while not hits and t.time() < deadline:
            t.sleep(0.02)
        t.sleep(0.2)
        assert hits == [("/api/generate", {"model": "qwen3.5:9b-16k"})]
    finally:
        srv.shutdown()


# --- tools: web search, grow status, routing ------------------------------------

def _post_stub(handler_fn):
    """HTTP stub; handler_fn(path, body_dict) -> (content_type, bytes), or
    (status, content_type, bytes)."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def _reply(self, body):
            out = handler_fn(self.path, body)
            status, (ctype, data) = (out[0], out[1:]) if len(out) == 3 else (200, out)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self._reply(json.loads(self.rfile.read(n) or b"{}"))

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


def _sse(*chunks):
    return ("text/event-stream",
            "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n")


def test_year_goes_into_search_tool_and_queries():
    """qwen3.5:9b searched 'who won the Masters this year 2024' in 2026."""
    import time as t

    m = _load_chat()
    desc = m.tools_for(t.strptime("2026-09-29", "%Y-%m-%d"))[0]["function"]["description"]
    assert "It is 2026" in desc and "{year}" not in desc
    assert "{year}" in m.TOOLS[0]["function"]["description"]   # template untouched


def test_routing_patterns():
    m = _load_chat()
    for q in ["Who won the World Cup this year?", "Who is the chair of the Federal Reserve?",
              "what's the latest on the storm", "price of bitcoin"]:
        assert m.CURRENT_WORDS.search(q), q
    for q in ["What is 17 times 3?", "Tell me a joke", "How does a heat pump work?"]:
        assert not m.CURRENT_WORDS.search(q), q
    for q in ["How is the reservoir doing?", "what's the humidity in the tent", "EC check"]:
        assert m.GROW_WORDS.search(q), q
    assert not m.GROW_WORDS.search("What is 17 times 3?")


def test_web_search_formats_and_reports_outages():
    m = _load_chat()
    state = {"mode": "ok"}

    def handler(path, _body):
        if state["mode"] == "ok":
            d = {"answers": [{"answer": "Spain"}],
                 "infoboxes": [{"infobox": "2026 FIFA World Cup", "content": "Held in NA."}],
                 "results": [{"title": f"T{i}", "url": f"https://x/{i}", "content": "snip " * 3}
                             for i in range(10)]}
        else:
            d = {"results": [], "unresponsive_engines": [["brave", "Suspended: too many requests"]]}
        return "application/json", json.dumps(d).encode()

    base, srv = _post_stub(handler)
    try:
        m.SEARX_URL = base
        out = m.web_search("world cup 2026")
        assert out.splitlines()[0] == "Answer: Spain"
        assert "Infobox (2026 FIFA World Cup)" in out
        assert "8. T7 - https://x/7" in out and "T8" not in out
        state["mode"] = "down"
        out = m.web_search("anything")
        assert out.startswith("SEARCH UNAVAILABLE") and "brave: Suspended" in out
    finally:
        srv.shutdown()


def test_stream_reply_collects_tool_calls_both_shapes():
    m = _load_chat()
    whole = _sse({"choices": [{"delta": {"content": "", "tool_calls": [
        {"id": "c1", "index": 0, "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":"q"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    fragments = _sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2",
                                                "function": {"name": "grow_status", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]})
    for body, want in ((whole, ("c1", "web_search", '{"query":"q"}')),
                       (fragments, ("c2", "grow_status", "{}"))):
        base, srv = _post_stub(lambda p, b, body=body: body)
        try:
            m.LLM_URL = base
            calls = []
            reply, _, _ = m.stream_reply("m", [], "ollama", tools=m.TOOLS, calls_out=calls)
            assert reply == ""
            assert (calls[0]["id"], calls[0]["name"], calls[0]["arguments"]) == want
        finally:
            srv.shutdown()


def test_chat_turn_runs_a_tool_round():
    """Model asks for a search, gets the result, then answers from it."""
    m = _load_chat()
    seen = []

    def handler(path, body):
        if path.startswith("/search"):
            return "application/json", json.dumps(
                {"results": [{"title": "Golden Tempo wins", "url": "https://w", "content": "Derby"}]}).encode()
        seen.append(body)
        if len(seen) == 1:
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": "c1", "index": 0, "function": {"name": "web_search",
                                                      "arguments": '{"query":"derby 2026"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "Golden Tempo won."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        # No current-events words here, so no auto-search: this exercises the
        # model-driven tool round.
        reply, _, used, grow = m.chat_turn("m", hist, "Tell me about the derby horse",
                                           "ollama", True, log=lambda *a: None)
        assert reply == "Golden Tempo won."
        assert used == [{"name": "web_search", "args": {"query": "derby 2026"}}]
        assert [h["role"] for h in hist] == ["system", "user", "assistant", "tool"]
        assert "Golden Tempo wins" in hist[-1]["content"]
        assert "tools" in seen[0] and "It is " in seen[0]["tools"][0]["function"]["description"]
        assert grow is False
    finally:
        srv.shutdown()


def test_chat_turn_answers_when_out_of_tool_rounds():
    """Qwen 3.8 on llama-server (2026-10-16) kept searching; once tools were no
    longer offered it wrote the call as text, and npu-chat printed and spoke
    '<tool_call> <function=web_search> ...' as the answer."""
    m = _load_chat()
    seen = []
    raw = ("<tool_call>\n<function=web_search>\n<parameter=query>\nalan xtra\n"
           "</parameter>\n</function>\n</tool_call>")

    def handler(path, body):
        if path.startswith("/search"):
            return "application/json", json.dumps(
                {"results": [{"title": "piper voices", "url": "https://p", "content": "alan: low, medium"}]}).encode()
        seen.append(body)
        if "tools" in body:
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": f"c{len(seen)}", "index": 0, "function": {"name": "web_search",
                                                                "arguments": '{"query":"alan xtra"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "Alan comes in low and medium. " + raw}}]})

    base, srv = _post_stub(handler)
    spoken = []
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        reply, _, used, _ = m.chat_turn("m", hist, "Is there an extra version of the alan voice?",
                                        "llama.cpp", True, on_delta=spoken.append,
                                        log=lambda *a: None)
        assert len(seen) == m.MAX_TOOL_ROUNDS + 1 and "tools" not in seen[-1]
        assert "answer now" in seen[-1]["messages"][-1]["content"].lower()
        assert reply == "Alan comes in low and medium."
        assert "<tool_call>" not in "".join(spoken)
        assert "answer now" not in json.dumps(hist).lower()    # the nudge is not kept
    finally:
        srv.shutdown()


def test_current_question_forces_a_search_the_model_writes():
    """Left to decide, the model answered 'Jerome Powell' 3/3 without searching,
    so a current question still gets a search, but the model writes the query:
    the user's own sentence used to be sent (2026-10-17, 6 of 22 searches)."""
    import urllib.parse as up

    m = _load_chat()
    queries, seen = [], []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(up.parse_qs(up.urlsplit(path).query)["q"][0])
            return "application/json", json.dumps(
                {"results": [{"title": "Warsh sworn in", "url": "https://f", "content": "chair"}]}).encode()
        seen.append(body)
        if body.get("tool_choice") == "required":
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": "c1", "index": 0, "function": {"name": "web_search",
                                                      "arguments": '{"query":"Federal Reserve chair 2026"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "Kevin Warsh."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        question = "Who is the chair of the Federal Reserve? Asking for my aunt Linda."
        reply, _, used, _ = m.chat_turn("m", hist, question, "llama.cpp", True,
                                        log=lambda *a: None)
        assert queries == ["Federal Reserve chair 2026"]          # the model's words only
        assert [t["function"]["name"] for t in seen[0]["tools"]] == ["web_search"]
        assert "tool_choice" not in seen[1] and "read_page" in json.dumps(seen[1]["tools"])
        assert used == [{"name": "web_search", "args": {"query": "Federal Reserve chair 2026"},
                         "forced": True}]
        assert hist[1]["content"] == question
        assert [h["role"] for h in hist] == ["system", "user", "assistant", "tool"]
        assert reply == "Kevin Warsh."
    finally:
        srv.shutdown()


def test_search_off_sends_nothing_to_the_search_engines():
    """User 2026-10-17: keep the questions local. With search off a current
    question is not searched by npu-chat, web_search is not offered, and a
    call the model writes anyway is answered without any request."""
    m = _load_chat()
    queries, seen, spoken = [], [], []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(path)
            return "application/json", b'{"results": []}'
        seen.append(body)
        if len(seen) == 1:
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": "c1", "index": 0, "function": {"name": "web_search",
                                                      "arguments": '{"query":"fed chair 2026"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "I can't check that."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        question = "Who is the chair of the Federal Reserve right now?"
        reply, _, used, _ = m.chat_turn("m", hist, question, "llama.cpp", True,
                                        on_delta=spoken.append, log=lambda *a: None,
                                        search_on=False)
        assert queries == []
        offered = [t["function"]["name"] for t in seen[0]["tools"]]
        assert "web_search" not in offered and "read_page" not in offered
        assert "grow_status" in offered and "tool_choice" not in seen[0]
        assert hist[1]["content"] == question                  # no results pasted in
        assert used == [{"name": "web_search", "args": {"query": "fed chair 2026"},
                         "result": "not sent: web search is off"}]
        assert hist[-1]["role"] == "tool" and hist[-1]["content"] == m.SEARCH_OFF
        assert "Let me check" not in "".join(spoken)
        assert reply == "I can't check that."
    finally:
        srv.shutdown()


def test_search_off_comes_from_the_environment_and_reaches_the_prompt(monkeypatch):
    monkeypatch.setenv("NPU_CHAT_WEB_SEARCH", "off")
    assert _load_chat().WEB_SEARCH is False
    monkeypatch.delenv("NPU_CHAT_WEB_SEARCH")
    m = _load_chat()
    assert m.WEB_SEARCH is True                                 # unchanged by default
    on, off = m.tools_hint("Saturday October 17, 2026"), m.tools_hint("Saturday October 17, 2026", False)
    assert "web_search" in on and "read_page" in on
    assert "web_search" not in off and "read_page" not in off
    assert "cannot look anything up" in off and "grow_status" in off and "October 17" in off
    assert [t["function"]["name"] for t in m.tools_for(search=False)] == ["grow_status", "grow_control"]


def test_forced_search_asks_again_before_saying_an_unsearched_answer():
    """2026-10-18 eval: once in 24, llama-server let Qwen answer "the 2026 World
    Cup hasn't happened yet" despite tool_choice "required"."""
    import urllib.parse as up

    m = _load_chat()
    queries, spoken = [], []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(up.parse_qs(up.urlsplit(path).query)["q"][0])
            return "application/json", json.dumps(
                {"results": [{"title": "Spain beat Argentina", "url": "https://f", "content": "final"}]}).encode()
        if body.get("tool_choice") != "required":
            return _sse({"choices": [{"delta": {"content": "Spain won."}}]})
        if body["messages"][-1] == m.SEARCH_FIRST:
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": "c1", "index": 0, "function": {"name": "web_search",
                                                      "arguments": '{"query":"World Cup 2026 winner"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "The 2026 World Cup hasn't happened yet."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        reply, _, used, _ = m.chat_turn("m", hist, "Who won the World Cup this year?", "llama.cpp",
                                        True, on_delta=spoken.append, log=lambda *a: None)
        assert queries == ["World Cup 2026 winner"] and reply == "Spain won."
        assert "happened" not in "".join(spoken)
        assert m.SEARCH_FIRST not in hist
    finally:
        srv.shutdown()


def test_this_year_means_this_year_in_the_query():
    """Qwen 3.8 searched '2025 Stanley Cup winner' for 'this year' in 2026."""
    m = _load_chat()
    q = lambda args, said: json.loads(m.query_for_this_year(json.dumps(args), said, 2026))["query"]
    assert q({"query": "2025 Stanley Cup winner"}, "Who won the Stanley Cup this year?") == \
        "2026 Stanley Cup winner"
    assert q({"query": "Stanley Cup winner"}, "Who won the Stanley Cup this year?") == \
        "Stanley Cup winner 2026"
    assert q({"query": "NBA 2025-26 standings"}, "How is the NBA going this year?").startswith(
        "NBA 2025-26 standings")                                   # a season is left alone
    assert q({"query": "2026 Masters winner"}, "Who won the Masters this year?") == "2026 Masters winner"
    assert q({"query": "2025 Stanley Cup winner"}, "Who won the Stanley Cup last year?") == \
        "2025 Stanley Cup winner"
    assert q({"query": "2024 Stanley Cup winner"}, "Who won it in 2024, this year or not?") == \
        "2024 Stanley Cup winner"                                  # the user named a year


_PAGE = """<html><head><title>2026 Kentucky Derby - Racing</title>
<script>var trackers = "Golden Tempo derby ad script";</script></head><body>
<nav><a href="/">Home</a> <a href="/racing">Racing news and the Kentucky Derby archive</a></nav>
<div>Subscribe to our newsletter for daily racing tips and offers from partners.</div>
<p>The 152nd Kentucky Derby was run on May 2, 2026. Golden Tempo, ridden by Jose Ortiz,
won by a length over Renegade.</p>
<footer>Copyright 2026 Racing Media, all rights reserved, Kentucky Derby is a trademark.</footer>
</body></html>"""


def _page_stub(npu_answer):
    """One server: GET /page is the web page, POST /v1/chat/completions the NPU."""
    asked = []

    def handler(path, body):
        if path == "/page":
            return "text/html; charset=utf-8", _PAGE.encode()
        asked.append(body)
        if npu_answer is None:
            return 500, "text/plain", b"npu busy"
        return "application/json", json.dumps(
            {"choices": [{"message": {"content": npu_answer}}]}).encode()

    base, srv = _post_stub(handler)
    return base, srv, asked


def test_read_page_hands_the_npu_only_the_relevant_text():
    """Like WebFetch: the small model on the NPU reads the page, and only its
    short answer comes back for the conversation."""
    m = _load_chat()
    base, srv, asked = _page_stub("Golden Tempo won, ridden by Jose Ortiz.")
    try:
        m.URL = base
        m._public_host = lambda host: True          # the stub is on 127.0.0.1
        found = [{"title": "Derby", "url": f"{base}/page", "content": "snippet"}]
        out = m.read_page(f"{base}/page", "Who won the 2026 Kentucky Derby?", found=found)
        assert out == f"From {base}/page:\nGolden Tempo won, ridden by Jose Ortiz."
        assert asked[0]["model"] == m.PAGE_MODEL and asked[0]["stream"] is False
        page_text = asked[0]["messages"][1]["content"]
        assert "won by a length over Renegade" in page_text
        assert "ad script" not in page_text and "Home" not in page_text   # script, menu
        assert found[-1] == {"title": "2026 Kentucky Derby - Racing", "url": f"{base}/page",
                             "content": "Golden Tempo won, ridden by Jose Ortiz."}
    finally:
        srv.shutdown()


def test_read_page_falls_back_to_the_best_passage_when_the_npu_fails():
    m = _load_chat()
    base, srv, asked = _page_stub(None)
    try:
        m.URL = base
        m._public_host = lambda host: True
        found = [{"title": "Derby", "url": f"{base}/page", "content": "snippet"}]
        out = m.read_page(f"{base}/page", "Who won the 2026 Kentucky Derby?", found=found)
        assert len(asked) == 1 and "Golden Tempo, ridden by Jose Ortiz" in out
        assert len(out) <= len(f"From {base}/page:\n") + 800 and "ad script" not in out
    finally:
        srv.shutdown()


def test_read_page_reads_only_this_turns_results_on_public_hosts():
    """A URL the model writes itself could carry the conversation out in its
    query string; a result could point at this box or the LAN."""
    m = _load_chat()
    base, srv, asked = _page_stub("x")
    try:
        m.URL = base
        out = m.read_page(f"{base}/page?q=my+secret", "q", found=[{"url": f"{base}/page"}])
        assert out.startswith("Not read") and asked == []
        out = m.read_page(f"{base}/page", "q", found=[{"url": f"{base}/page"}])
        assert out.startswith("Not read") and asked == []          # 127.0.0.1 is not public
    finally:
        srv.shutdown()
    for host in ("127.0.0.1", "10.0.0.2", "192.168.0.10", "localhost", "printer.local", None):
        assert not m._public_host(host), host
    assert m._public_host("1.1.1.1")
    assert not m._public_url("ftp://1.1.1.1/x") and not m._public_url("file:///etc/passwd")


def test_a_failed_turn_says_why_instead_of_going_silent():
    """2026-10-17: a session that filled the 32K context answered nothing, in
    silence."""
    m = _load_chat()
    full = json.dumps({"error": {"code": 400, "type": "exceed_context_size_error", "message":
                                 "request (32803 tokens) exceeds the available context size "
                                 "(32768 tokens), try increasing it"}}).encode()
    base, srv = _post_stub(lambda path, body: (400, "application/json", full))
    try:
        m.LLM_URL = base
        reply, _, _ = m.stream_reply("m", [{"role": "user", "content": "hi"}], "llama.cpp")
        assert reply == "" and m.LAST_LLM_ERROR["kind"] == "full"
        assert "32803 tokens" in m.LAST_LLM_ERROR["detail"]
        assert m.failure_notice(m.LAST_LLM_ERROR) == \
            "This conversation has filled my memory, so I can't answer."
    finally:
        srv.shutdown()
    m.LLM_URL = "http://127.0.0.1:9"           # nothing listens there
    m.stream_reply("m", [{"role": "user", "content": "hi"}], "llama.cpp")
    assert m.failure_notice(m.LAST_LLM_ERROR) == "I can't reach the model right now."
    assert m.failure_notice({}) == "I didn't get an answer that time."


def test_grow_status_reads_cached_bar_json(tmp_path):
    """The bar JSON has raw newlines inside strings; strict=False must take it."""
    m = _load_chat()
    for cmd in ("acinf", "vivosun", "yinmik"):
        f = tmp_path / cmd
        f.write_text("#!/bin/sh\nprintf '{\"text\": \"t\", \"tooltip\": \"%s:\\n  line2\"}\\n'\n" % cmd)
        f.chmod(0o755)
    old = os.environ["PATH"]
    os.environ["PATH"] = f"{tmp_path}:{old}"
    try:
        out = m.grow_status()
    finally:
        os.environ["PATH"] = old
    assert out.splitlines() == ["acinf:", "  line2", "vivosun:", "  line2", "yinmik:", "  line2"]


def test_resume_picks_latest_chat_session_and_skips_the_rest(tmp_path):
    import time as t

    m = _load_chat()
    chat = tmp_path / "a.jsonl"
    chat.write_text("\n".join(json.dumps(r) for r in [
        {"via": "text", "user": "hi", "assistant": "hello"},
        {"via": "voice-discarded", "user": "Thank you.", "assistant": None, "sent": False},
        {"via": "vad-discarded", "user": None, "assistant": None, "sent": False},
        {"via": "voice", "user": "17x3?", "assistant": "51"}]) + "\n")
    t.sleep(0.01)
    agent = tmp_path / "b.jsonl"   # newer, but an agent session
    agent.write_text(json.dumps({"user": "fix", "assistant": "done", "session_id": "s"}) + "\n")
    assert m.latest_chat_log(str(tmp_path)) == str(chat)
    assert m.history_from_log(str(chat)) == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "17x3?"}, {"role": "assistant", "content": "51"}]
    assert m.latest_chat_log(str(tmp_path / "empty")) is None


# --- npu-vad (the speech check npu-dictate uses) ---------------------------------

def test_npu_vad_exit_codes_and_parity_with_npu_chat(tmp_path):
    """Silence exits 1, speech exits 0, and the peak matches npu-chat's gate."""
    import pytest

    m = _load_chat()
    silence = _wav16k(tmp_path / "silence.wav", [0] * 32000)
    r = subprocess.run([sys.executable, "npu-vad", silence], capture_output=True, text=True)
    if r.returncode == 2:
        pytest.skip("Silero VAD model or onnxruntime not available")
    assert r.returncode == 1 and float(r.stdout) < m.VAD_MIN
    assert abs(float(r.stdout) - m.vad_peak(silence)) < 1e-3
    voice = m.find_voice(m.DEFAULT_VOICE)
    if not (voice and shutil.which("piper-tts")):
        pytest.skip("no piper voice to synthesise speech")
    raw, speech = tmp_path / "raw.wav", tmp_path / "speech.wav"
    subprocess.run(["piper-tts", "-m", voice, "-f", str(raw)], input=b"Type this sentence.",
                   capture_output=True, check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
                    "-ar", "16000", "-ac", "1", str(speech)], check=True)
    r = subprocess.run([sys.executable, "npu-vad", str(speech)], capture_output=True, text=True)
    assert r.returncode == 0
    assert abs(float(r.stdout) - m.vad_peak(str(speech))) < 1e-3


def test_npu_vad_reports_unavailable_without_model(tmp_path):
    env = dict(os.environ, NPU_CHAT_VAD_MODEL=str(tmp_path / "missing.onnx"))
    r = subprocess.run([sys.executable, "npu-vad", _wav16k(tmp_path / "s.wav", [0] * 16000)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and r.stdout.strip() == "unavailable"


def test_dictate_uses_the_speech_check():
    src = open("npu-dictate").read()
    assert "npu-vad" in src and "no speech detected" in src


# --- spoken sources ---------------------------------------------------------------

def test_site_names():
    m = _load_chat()
    assert m.site_name("https://en.wikipedia.org/wiki/X", "2026 NBA Finals - Wikipedia") == "Wikipedia"
    assert m.site_name("https://www.fifa.com/a", "World Cup 2026: standings") == "FIFA"
    assert m.site_name("https://www.espn.com/x", "NFL History") == "ESPN"
    assert m.site_name("https://www.basketball-reference.com/p",
                       "Finals | Basketball-Reference.com") == "Basketball-Reference"
    assert m.site_name("https://www.bbc.co.uk/sport/x", "") == "BBC"
    assert m.site_name("https://www.abc.net.au/news/x", "") == "ABC"
    assert m.site_name("https://www.masters.com/en_US/scores",
                       "2026 Masters Tournament - Final Leaderboard") == "Masters"
    assert m.site_name("https://www.nfl.com/news/x", "Seahawks beat Patriots 29 - 13") == "NFL"
    assert m.site_name("https://www.bbc.co.uk/news/x", "Storm hits - BBC News") == "BBC News"


def test_attribution_finds_support_and_flags_inventions():
    """The fabricated 'Chiefs 61-9, reported by ESPN' matched none of the results."""
    m = _load_chat()
    found = [
        {"title": "Super Bowl winners", "url": "https://betnow.eu/x",
         "content": "The most recent trophy belongs to Seattle, a 29-13 job on New England."},
        {"title": "Seahawks beat Patriots", "url": "https://profootballmania.com/x",
         "content": "The Seattle Seahawks beat the New England Patriots 29-13 in Super Bowl LX."}]
    src = m.attribute("The Seattle Seahawks won Super Bowl LX, beating the Patriots 29-13.", found)
    assert src["url"] == "https://profootballmania.com/x" and src["site"] == "Profootballmania"
    assert m.attribute("The Kansas City Chiefs beat the San Francisco 49ers 61-9, "
                       "as reported by ESPN.", found) is None
    assert m.attribute("", found) is None


# --- grow control: validated, confirmed, then run ----------------------------------

_ACINF = "Infinity 69 Pro: 74F\n  1. Port 1: on @ 3/10\n  4. Exhaust: schedule @ 7/10  (2h left)"
_VIVO = "GrowHub E42A: 68F\n  light 100% · duct fan 3/10 · circ fan 3/10"


def test_plan_control_validates_and_states_mode_changes():
    import pytest

    m = _load_chat()
    argv, q = m.plan_control("vivosun", "circ", "5", _ACINF, _VIVO)
    assert argv == ["vivosun", "set", "circ", "5"]
    assert "3/10 now" in q and "manual mode" in q
    argv, q = m.plan_control("acinfinity", "port 4", "6", _ACINF, _VIVO)
    assert argv == ["acinf", "set", "4", "6"]
    assert "schedule mode" in q and "ends its schedule mode" in q
    argv, q = m.plan_control("acinfinity", "1", "0", _ACINF, _VIVO)
    assert argv == ["acinf", "set", "1", "0"] and "turn it off" in q and "ends" not in q
    assert m.plan_control("acinfinity", "4", "schedule", _ACINF, _VIVO)[0] == \
        ["acinf", "mode", "4", "schedule"]
    assert m.plan_control("vivosun", "light", "0", _ACINF, _VIVO)[0][-1] == "0"
    for bad in [("vivosun", "light", "20"), ("vivosun", "duct", "11"), ("vivosun", "pump", "1"),
                ("acinfinity", "7", "3"), ("acinfinity", "2", "turbo"), ("nest", "x", "1"),
                ("vivosun", "circ", "5; rm -rf ~")]:
        with pytest.raises(ValueError):
            m.plan_control(*bad, acinf_tip=_ACINF, vivo_tip=_VIVO)


def test_plan_control_rejects_non_whole_numbers():
    import pytest

    m = _load_chat()
    for v in ("inf", "1e999", "0.6", "nan", "-1", "5.5"):
        for device, target in (("vivosun", "duct"), ("acinfinity", "4")):
            with pytest.raises(ValueError):
                m.plan_control(device, target, v, "", "")
    assert m.plan_control("vivosun", "duct", "5.0", "", "")[0] == ["vivosun", "set", "duct", "5"]
    assert m.plan_control("vivosun", "light", "50%", "", "")[0][-1] == "50"

    def never(_argv):
        raise AssertionError("run_control must not be reached")
    m.run_control = never
    m._cached_tooltip = lambda c: ""
    result = m.run_tool("grow_control", '{"device":"vivosun","target":"duct","value":"inf"}',
                        confirm=lambda q: True)[1]
    assert result.startswith("not allowed"), result


def test_plan_control_warns_when_mode_is_unknown():
    m = _load_chat()
    q = m.plan_control("acinfinity", "4", "6", acinf_tip="", vivo_tip="")[1]
    assert "could not be read" in q and "makes it manual" in q


def test_grow_control_never_runs_without_a_yes(tmp_path):
    m = _load_chat()
    log = tmp_path / "ran.txt"
    for cmd in ("vivosun", "acinf"):
        f = tmp_path / cmd
        f.write_text(f'#!/bin/sh\nif [ "$1" = bar ]; then echo \'{{"tooltip": ""}}\'; exit 0; fi\n'
                     f'echo "{cmd} $*" >> {log}\necho "circ fan -> 5/10 confirmed"\n')
        f.chmod(0o755)
    old = os.environ["PATH"]
    os.environ["PATH"] = f"{tmp_path}:{old}"
    try:
        raw = '{"device": "vivosun", "target": "circ", "value": "5"}'
        asked = []
        assert "cancelled" in m.run_tool("grow_control", raw)[1]              # no confirm at all
        assert "cancelled" in m.run_tool("grow_control", raw,
                                         confirm=lambda q: asked.append(q) or False)[1]
        assert not log.exists(), "nothing may run before a yes"
        assert asked and asked[0].startswith("Set the circulation fan to 5")
        assert "not allowed" in m.run_tool(
            "grow_control", '{"device": "vivosun", "target": "light", "value": "5"}',
            confirm=lambda q: True)[1]
        assert not log.exists()
        out = m.run_tool("grow_control", raw, confirm=lambda q: True)[1]
        assert out == "Done. circ fan -> 5/10 confirmed."  # control_outcome wording
        assert log.read_text().strip() == "vivosun set circ 5"
    finally:
        os.environ["PATH"] = old


def test_attribution_ignores_words_from_the_question():
    """Every result about Ohio State says 'Ohio State'; that is not evidence of
    the opponent. The wrong 'Indiana' must not be credited to a source."""
    m = _load_chat()
    q = "Who did Ohio State play in their first football game this season?"
    found = [{"title": "Ohio State Buckeyes 2026 schedule", "url": "https://espn.com/x",
              "content": "Ohio State football first game of the 2026 season at Ohio Stadium."},
             {"title": "2026 Football Schedule - Ohio State", "url": "https://ohiostatebuckeyes.com/s",
              "content": "Sep. 05 vs Ball State. Columbus, Ohio; Sep. 12 at Texas."}]
    assert m.attribute("Ohio State played Indiana in their first game.", found, q) is None
    src = m.attribute("Ohio State opened against Ball State on September 5.", found, q)
    assert src and src["url"] == "https://ohiostatebuckeyes.com/s"


def test_grow_readings_never_reach_the_search_engines():
    """Regression: routing matched the attached readings ("...they are
    current") and sent the user's sentence plus all grow readings out as a
    public web query. Only the user's own words may be matched or sent."""
    m = _load_chat()
    queries = []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(path)
            return "application/json", b'{"results": []}'
        return _sse({"choices": [{"delta": {"content": "Fans are at 3/10."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        m.grow_status = lambda: "circ fan 3/10, pH 5.79 - current readings, right now"
        hist = [{"role": "system", "content": "s"}]
        m.chat_turn("m", hist, "Turn the circulation fan up to 5.", "ollama", True,
                    log=lambda *a: None)
        assert queries == []
        assert "pH 5.79" in hist[1]["content"]          # readings still reach the model
    finally:
        srv.shutdown()


def test_parse_control_phrasings():
    m = _load_chat()
    cases = {
        "Turn the circulation fan up to 5.": ("vivosun", "circ", "5"),
        "Set AC Infinity port 4 to 6.": ("acinfinity", "4", "6"),
        "Turn the grow light off.": ("vivosun", "light", "0"),
        "Put the duct fan on 4.": ("vivosun", "duct", "4"),
        "Set the light to 50 percent": ("vivosun", "light", "50"),
        "turn port two off": ("acinfinity", "2", "off"),
        "switch port 4 to schedule": ("acinfinity", "4", "schedule"),
        "set the circ fan to natural": ("vivosun", "circ", "natural"),
        "turn the circ fan to five": ("vivosun", "circ", "5"),
        "Can you set the duct fan to 4?": ("vivosun", "duct", "4"),
        "Okay, turn the grow light off.": ("vivosun", "light", "0"),
        "Um, turn the light off.": ("vivosun", "light", "0"),
        "set the circ fan to 5 out of 10": ("vivosun", "circ", "5"),
    }
    for text, want in cases.items():
        assert m.parse_control(text) == want, text
    for text in ["What is the duct fan at?", "turn up the fans", "Turn off the TV",
                 "set the light and the duct fan to 4", "turn the circ fan up by 2",
                 # timed, conditional or a question: these go to the model
                 "turn port 2 off for 10 minutes",
                 "turn the light off in 30 minutes",
                 "How high should I set the light? It's at 75 percent",
                 "Should I turn the duct fan to 5 for flowering?",
                 "set the light to 1000",
                 "turn the light off at 6 pm",
                 "turn the lights off at 8am",
                 "turn the light off at 8:30",
                 "turn off the light for a while",
                 "turn the light off at noon"]:
        assert m.parse_control(text) is None, text
    assert m.control_intent("turn up the fans") and not m.control_intent("what is the pH")


def test_control_intent_ignores_questions():
    m = _load_chat()
    for text in ["How high should I set the light?", "Why do the fans turn off at night?",
                 "Should I turn the duct fan to 5 for flowering?"]:
        assert not m.control_intent(text), text
    for text in ["turn up the fans", "Can you turn the fans up?"]:
        assert m.control_intent(text), text


def test_parsed_control_skips_the_model_and_reports_what_happened(tmp_path):
    """The model once said 'I'll increase your circulation fan to 5 right now'
    and called nothing. A parsed request never goes to the model."""
    m = _load_chat()
    llm_calls = []

    def handler(path, body):
        llm_calls.append(path)
        return _sse({"choices": [{"delta": {"content": "I'll do that right now."}}]})

    base, srv = _post_stub(handler)
    log = tmp_path / "ran.txt"
    f = tmp_path / "vivosun"
    f.write_text(f'#!/bin/sh\nif [ "$1" = bar ]; then echo \'{{"tooltip": "circ fan 3/10"}}\'; exit 0; fi\n'
                 f'echo "$*" >> {log}\necho "circ fan -> 5/10 confirmed"\n')
    f.chmod(0o755)
    old = os.environ["PATH"]
    os.environ["PATH"] = f"{tmp_path}:{old}"
    try:
        m.LLM_URL = base
        for answer, want in ((False, "Cancelled. Nothing was changed."),
                             (True, "Done. circ fan -> 5/10 confirmed.")):
            hist = [{"role": "system", "content": "s"}]
            reply, _, used, _ = m.chat_turn("m", hist, "Turn the circulation fan up to 5.",
                                            "ollama", True, log=lambda *a: None,
                                            confirm=lambda q, a=answer: a)
            assert reply == want
            assert used[0]["parsed"] is True
        assert llm_calls == [], "a parsed control request must not reach the model"
        assert log.read_text().strip() == "set circ 5"          # ran once, after the yes
        hist = [{"role": "system", "content": "s"}]
        reply, _, _, _ = m.chat_turn("m", hist, "Set the light to 20 percent", "ollama", True,
                                     log=lambda *a: None, confirm=lambda q: True)
        assert reply.startswith("I can't do that: light must be 0") and "Nothing was changed" in reply
    finally:
        os.environ["PATH"] = old
        srv.shutdown()


def test_npu_vad_runtime_error_means_cannot_tell(tmp_path):
    import pytest
    pytest.importorskip("numpy")
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "onnxruntime.py").write_text(
        "class InferenceSession:\n"
        "    def __init__(self, *a, **k):\n        pass\n"
        "    def run(self, *a, **k):\n"
        "        raise RuntimeError('Invalid input name: state')\n")
    wav = tmp_path / "a.wav"
    _wav16k(wav, [0] * 16000)
    r = subprocess.run([sys.executable, "npu-vad", str(wav)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(fake),
                            "NPU_CHAT_VAD_MODEL": str(tmp_path / "any.onnx")})
    assert r.returncode == 2
    assert r.stdout.strip() == "unavailable"


def test_vad_runtime_error_turns_the_gate_off(tmp_path):
    import pytest
    pytest.importorskip("numpy")
    m = _load_chat()

    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("Invalid input name: state")
    m._VAD["session"] = Broken()
    wav = tmp_path / "a.wav"
    _wav16k(wav, [0] * 16000)
    assert m.vad_peak(str(wav)) is None
    # Off for the rest of the session: later captures do not pay for it again.
    assert m._VAD["session"] is False
    assert m.vad_peak(str(wav)) is None


def test_npu_vad_crash_outside_peak_means_cannot_tell(tmp_path):
    wav = tmp_path / "a.wav"
    _wav16k(wav, [0] * 16000)
    # A bad threshold used to raise at import: Python's exit 1, read as "no speech".
    r = subprocess.run([sys.executable, "npu-vad", str(wav)], capture_output=True, text=True,
                       env={**os.environ, "NPU_CHAT_VAD_MIN": "abc",
                            "NPU_CHAT_VAD_MODEL": str(tmp_path / "none.onnx")})
    assert r.returncode == 2


def test_sigint_keeps_keyboard_interrupt():
    import signal as sig
    m = _load_chat()
    saved = {s: sig.getsignal(s) for s in (sig.SIGINT, sig.SIGTERM, sig.SIGHUP)}
    try:
        m._HANDLERS_SET = False
        m._install_cleanup_handlers()
        assert sig.getsignal(sig.SIGINT) is saved[sig.SIGINT]
        assert sig.getsignal(sig.SIGTERM) == m._cleanup_and_die
    finally:
        for s, h in saved.items():
            sig.signal(s, h)


def test_nothing_changed_note_in_every_mode():
    m = _load_chat()
    for tools_on in (True, False):
        note = m.nothing_changed_note("set the duct fan to 5", "Done, I set it.", [], tools_on)
        assert "nothing was changed" in note
    assert m.nothing_changed_note("How high should I set the light?", "About 30 cm.", [],
                                  True) is None
    assert m.nothing_changed_note("set the duct fan to 5", "ok",
                                  [{"name": "grow_control"}], True) is None


def test_source_verdict():
    m = _load_chat()
    seattle = [{"title": "Seattle Seahawks win", "url": "https://nfl.com/x",
                "content": "The Seattle Seahawks beat the Patriots."}]
    # Numbers from the attached grow readings need no web source...
    assert m.source_verdict("The tent is at 62% humidity.", seattle,
                            "Is the tent too humid outside today?", "humidity 62% temp 74F") == \
        ("skip", None)
    # ...but a weather claim alongside them is still checked.
    assert m.source_verdict("It's 12 degrees and raining in Portland; the tent is at 62%.",
                            seattle, "What's the weather outside, and is the tent humid?",
                            "humidity 62%")[0] == "unsupported"
    assert m.source_verdict("Yes, it is raining.", seattle, "Is it raining?") == ("skip", None)
    assert m.source_verdict("Seattle's defense won it.", seattle, "Who won?")[0] == "ok"
    chiefs = [{"title": "Chiefs news", "url": "https://x.com/a",
               "content": "Fans in Kansas City gathered."}]
    assert m.source_verdict("The Kansas City Chiefs beat the Philadelphia Eagles 40-22.",
                            chiefs, "Who won the Super Bowl?")[0] == "unsupported"


def test_direct_answers_count_as_support():
    m = _load_chat()

    def handler(path, body):
        return "application/json", json.dumps(
            {"answers": [{"answer": "Spain"}],
             "results": [{"title": "World Cup", "url": "https://fifa.com/a",
                          "content": "The final was played in July."}]}).encode()
    base, srv = _post_stub(handler)
    try:
        m.SEARX_URL = base
        found = []
        m.web_search("world cup 2026", found=found)
    finally:
        srv.shutdown()
    assert m.source_verdict("Spain won the 2026 World Cup.", found,
                            "Who won the World Cup this year?")[0] == "ok"


def test_grow_questions_are_not_web_searched():
    m = _load_chat()
    queries = []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(path)
            return "application/json", b'{"results": []}'
        if body.get("tool_choice") == "required":
            return _sse({"choices": [{"delta": {"tool_calls": [
                {"id": "c1", "index": 0, "function": {"name": "web_search",
                                                      "arguments": '{"query":"weather today"}'}}]}}]})
        return _sse({"choices": [{"delta": {"content": "It is 5.8."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        m.grow_status = lambda: "pH 5.8"
        hist = [{"role": "system", "content": "s"}]
        out = m.chat_turn("m", hist, "What's the pH right now?", "ollama", True,
                          log=lambda *a: None)
        assert queries == []
        assert out[3] is True                       # the readings were attached
        m.chat_turn("m", hist, "What's the temperature outside today?", "ollama", True,
                    log=lambda *a: None)
        assert len(queries) == 1
    finally:
        srv.shutdown()


def test_control_reply_reflects_the_outcome():
    m = _load_chat()
    m._cached_tooltip = lambda c: ""
    unconfirmed = ("done: circ fan -> 5/10 sent and accepted by the cloud, but the controller "
                   "had not reported it within 12s")
    cases = [
        ("done: ", lambda r: r == "Done."),
        ("done: circ fan was already at 5/10 in manual mode; nothing changed",
         lambda r: r.startswith("Nothing needed changing:")),
        (unconfirmed, lambda r: r.startswith("Sent, but not confirmed yet:")),
        ("done: circ fan -> 5/10 (controller confirmed)",
         lambda r: r == "Done. circ fan -> 5/10 (controller confirmed)."),
    ]
    for result, ok in cases:
        m.run_control = lambda argv, result=result: result
        reply, _ = m.handle_control(("vivosun", "circ", "5"), lambda q: True)
        assert ok(reply), reply
        assert ". ." not in reply
    m.run_control = lambda argv: unconfirmed
    tool = m.run_tool("grow_control", '{"device":"vivosun","target":"circ","value":"5"}',
                      confirm=lambda q: True)[1]
    assert tool.startswith("Sent, but not confirmed yet:"), tool


def test_dictate_stop_waits_for_the_recorder(tmp_path):
    """Fully faked: no microphone, NPU server or focused window is touched."""
    import time as _t

    bin_ = tmp_path / "bin"
    bin_.mkdir()
    fakes = {
        # Python, not bash: a job started in the background by a non-interactive
        # shell inherits SIGINT as ignored, and bash cannot trap that, while a
        # real recorder (and Python) can still install its own handler.
        "pw-record": f"#!{sys.executable}\n"
                     "import signal, sys, time\n"
                     "out = sys.argv[-1]\n"
                     "open(out, 'wb').write(b'RIFF')\n"
                     "def stop(*_):\n"
                     "    time.sleep(0.5)\n"
                     "    open(out, 'ab').write(b'END')\n"
                     "    sys.exit(0)\n"
                     "signal.signal(signal.SIGINT, stop)\n"
                     "signal.signal(signal.SIGTERM, stop)\n"
                     "while True:\n"
                     "    time.sleep(0.05)\n",
        "ffmpeg": '#!/bin/bash\nwhile [ "$1" != "-i" ]; do shift; done\nsrc="$2"\n'
                  'cp "$src" "${@: -1}"\n',
        "npu-vad": "#!/bin/sh\necho unavailable\nexit 2\n",
        "whisper-npu": "#!/bin/sh\nexit 1\n",
        "wtype": "#!/bin/sh\nexit 0\n",
        "wl-copy": "#!/bin/sh\nexit 0\n",
        "notify-send": "#!/bin/sh\nexit 0\n",
    }
    for name, body in fakes.items():
        f = bin_ / name
        f.write_text(body)
        f.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    env = {"HOME": str(tmp_path), "PATH": f"{bin_}:/usr/bin:/bin",
           "XDG_RUNTIME_DIR": str(run), "XDG_STATE_HOME": str(tmp_path / "state"),
           "FLM_PORT": "9"}
    subprocess.run(["bash", "npu-dictate", "--no-notify", "start"], env=env, timeout=30)
    _t.sleep(0.3)
    subprocess.run(["bash", "npu-dictate", "--no-notify", "stop"], env=env, timeout=30)
    assert (run / "npu-dictate" / "rec.wav").read_bytes().endswith(b"END")


def test_cleanup_handlers_installed_even_when_a_thread_creates_the_dir():
    import signal as sg
    import threading
    m = _load_chat()
    saved = {s: sg.getsignal(s) for s in (sg.SIGTERM, sg.SIGHUP, sg.SIGINT)}
    try:
        t = threading.Thread(target=m.session_dir)
        t.start()
        t.join()
        m.session_dir()
        assert sg.getsignal(sg.SIGTERM) is m._cleanup_and_die
        assert sg.getsignal(sg.SIGHUP) is m._cleanup_and_die
    finally:
        for s, h in saved.items():
            sg.signal(s, h)
        shutil.rmtree(m._SESSION_DIR or "", ignore_errors=True)


def _flm_env(tmp_path, flm_body):
    """A temp state dir, port 9 and a fake `flm` first on PATH, so the real
    pidfile and server are never touched."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    flm = bin_ / "flm"
    flm.write_text(flm_body)
    flm.chmod(0o755)
    env = {**os.environ, "XDG_STATE_HOME": str(tmp_path / "state"), "FLM_PORT": "9",
           "PATH": f"{bin_}:{os.environ['PATH']}"}
    assert shutil.which("flm", path=env["PATH"]) == str(flm)
    pidfile = tmp_path / "state" / "flm" / "flm-server.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    return env, flm, pidfile


def test_stop_does_not_kill_an_unrelated_process(tmp_path):
    env, _, pidfile = _flm_env(tmp_path, "#!/bin/sh\nexit 1\n")
    victim = subprocess.Popen(["sleep", "30"])
    try:
        pidfile.write_text(str(victim.pid))
        r = subprocess.run(["bash", "whisper-npu", "--stop"], env=env, capture_output=True,
                           text=True, timeout=30)
        assert victim.poll() is None
        assert not pidfile.exists()
        assert "server not running" in r.stdout
    finally:
        victim.kill()
        victim.wait()


def test_stop_kills_flm_serve(tmp_path):
    import threading
    env, flm, pidfile = _flm_env(tmp_path, f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    proc = subprocess.Popen([str(flm), "serve"])
    threading.Thread(target=proc.wait, daemon=True).start()  # reap it, no zombie
    try:
        pidfile.write_text(str(proc.pid))
        r = subprocess.run(["bash", "whisper-npu", "--stop"], env=env, capture_output=True,
                           text=True, timeout=30)
        assert r.returncode == 0
        assert "server stopped" in r.stdout
        for _ in range(150):
            if proc.poll() is not None:
                break
            import time as _t
            _t.sleep(0.1)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_start_reports_an_early_exit(tmp_path):
    import time as _t
    env, _, pidfile = _flm_env(tmp_path, "#!/bin/sh\nexit 1\n")
    started = _t.monotonic()
    r = subprocess.run(["bash", "whisper-npu", "--start"], env=env, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode != 0
    assert _t.monotonic() - started < 10
    assert "exited" in r.stderr + r.stdout
    assert not pidfile.exists()


def test_flm_serve_does_not_inherit_the_lock(tmp_path):
    mark = tmp_path / "fd9"
    env, _, pidfile = _flm_env(
        tmp_path, f'#!/bin/sh\nif [ -e /proc/self/fd/9 ]; then echo held; else echo free; fi'
                  f' > "{mark}"\nexit 1\n')
    subprocess.run(["bash", "whisper-npu", "--start"], env=env, capture_output=True,
                   text=True, timeout=30)
    assert mark.read_text().strip() == "free"

