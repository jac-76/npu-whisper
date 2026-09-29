#!/usr/bin/env python3
"""Stand-in for `claude -p --input-format stream-json` in tests.

Speaks the event shapes captured from Claude Code 2.1.284 (see
fixtures/claude-stream.jsonl). Each user line gets: init, a Read tool call,
its result, a text reply echoing the message, and a result event.

Messages with special meaning: "hang" never answers, "die" exits mid-turn.
Every invocation's argv is appended to $FAKE_CLAUDE_LOG as a JSON line.
"""
import json
import os
import sys
import time

log = os.environ.get("FAKE_CLAUDE_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps({"argv": sys.argv[1:], "pid": os.getpid(),
                             "base_url": os.environ.get("ANTHROPIC_BASE_URL")}) + "\n")

sid = "sess-1"
if "--resume" in sys.argv:
    sid = sys.argv[sys.argv.index("--resume") + 1]


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)["message"]["content"]
    if msg == "hang":
        time.sleep(3600)
    if msg == "die":
        sys.stderr.write("fake: model load failed\n")
        sys.exit(3)
    emit({"type": "system", "subtype": "init", "session_id": sid, "tools": ["Bash", "Edit", "Read"]})
    emit({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 3, "session_id": sid})
    emit({"type": "assistant", "session_id": sid, "message": {"content": [
        {"type": "tool_use", "id": "call_1", "name": "Read",
         "input": {"file_path": os.path.join(os.getcwd(), "tally/config.py")}}]}})
    emit({"type": "user", "session_id": sid, "message": {"content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "DEFAULT_PORT = 7341"}]}})
    emit({"type": "assistant", "session_id": sid, "message": {"content": [
        {"type": "text", "text": f"echo: {msg}"}]}})
    emit({"type": "result", "subtype": "success", "is_error": False,
          "result": f"echo: {msg}", "session_id": sid, "num_turns": 2,
          "duration_ms": 12, "permission_denials": []})
