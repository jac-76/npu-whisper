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
    """HTTP stub; handler_fn(path, body_dict) -> (content_type, bytes)."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def _reply(self, body):
            ctype, data = handler_fn(self.path, body)
            self.send_response(200)
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
    assert m.search_query_for("Who won the Masters this year?", 2026) == \
        "Who won the Masters this year? 2026"
    assert m.search_query_for("Who won the 2025  Masters?", 2026) == "Who won the 2025 Masters?"


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


def test_chat_turn_searches_current_questions_itself():
    """Left to decide, the model answered 'Jerome Powell' 3/3 without searching."""
    import time as t
    import urllib.parse as up

    m = _load_chat()
    queries = []

    def handler(path, body):
        if path.startswith("/search"):
            queries.append(up.parse_qs(up.urlsplit(path).query)["q"][0])
            return "application/json", json.dumps(
                {"results": [{"title": "Warsh sworn in", "url": "https://f", "content": "chair"}]}).encode()
        return _sse({"choices": [{"delta": {"content": "Kevin Warsh."}}]})

    base, srv = _post_stub(handler)
    try:
        m.LLM_URL = base
        m.SEARX_URL = base
        hist = [{"role": "system", "content": "s"}]
        reply, _, used, _ = m.chat_turn("m", hist, "Who is the chair of the Federal Reserve?",
                                        "ollama", True, log=lambda *a: None)
        assert len(queries) == 1 and queries[0].endswith(str(t.localtime().tm_year))
        assert used[0]["auto"] is True
        assert "Warsh sworn in" in hist[1]["content"]
    finally:
        srv.shutdown()


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

_ACINF = "Infinity 69 Pro: 74F\n  1. Port 1: on @ 3/10\n  4. Port 4: schedule @ 7/10  (2h left)"
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
        assert out == "done: circ fan -> 5/10 confirmed"
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
    }
    for text, want in cases.items():
        assert m.parse_control(text) == want, text
    for text in ["What is the duct fan at?", "turn up the fans", "Turn off the TV",
                 "set the light and the duct fan to 4", "turn the circ fan up by 2"]:
        assert m.parse_control(text) is None, text
    assert m.control_intent("turn up the fans") and not m.control_intent("what is the pH")


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
