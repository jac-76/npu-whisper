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
    assert "reasoning_effort" not in m.request_body("granite", h, "llama.cpp")


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


def test_confirm_flag_exists_for_chat_mode():
    out = subprocess.run([sys.executable, "npu-chat", "--help"],
                         capture_output=True, text=True, check=True)
    assert "--confirm" in out.stdout
