#!/usr/bin/env python3
"""Benchmark the LLM half of npu-chat against any OpenAI-compatible server.

Same prompts, same system prompt and same max_tokens as npu-chat, so the numbers
describe what a voice turn actually costs. Works against FastFlowLM on the NPU,
ollama, or llama-server -- which is the point: one script, comparable numbers.

Per model it reports:
  - time to first token and total wall time, warm (mean +/- stdev)
  - decode speed from the server's own usage count, when it sends one
  - whether reasoning text arrived (delta.reasoning / reasoning_content) and
    whether the visible reply came back empty -- the failure a thinking model
    produces in npu-chat, which reads only delta.content
  - a two-step word problem, N times (answer 7; gemma3:1b on the NPU got 0/5)
  - a tool-call probe: pass only if the server returns finish_reason=tool_calls
    with parseable arguments. A prose "weather report" is a FAIL: the model
    invented the tool's result, which is what FastFlowLM does.

Usage:
  bench_llm.py --no-think --url http://192.168.1.36:11434 --model qwen3.5:9b-16k
  bench_llm.py --url http://127.0.0.1:52625 --model qwen3.5:4b      # the NPU
"""

import argparse
import json
import re
import statistics
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROMPTS = {
    "factual": "What is the capital of Ohio, and one thing it is known for?",
    "arith": "What is 17 times 3?",
}
APPLES = "I have 5 apples, eat 2, then buy 4 more. How many do I have?"
APPLES_RIGHT = re.compile(r"\b(7|seven)\b", re.I)
TOOL_PROMPT = "What's the weather in Columbus, Ohio right now?"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]


def chat_system_prompt() -> str:
    """npu-chat's own system prompt, loaded from the script so they cannot drift."""
    mod = types.ModuleType("npu_chat")
    src = (HERE / "npu-chat").read_text()
    exec(compile(src, "npu-chat", "exec"), mod.__dict__)
    return mod.SYSTEM_PROMPT


def post(url: str, body: dict, timeout: float = 300):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_once(base: str, model: str, system: str, prompt: str,
                extra: dict | None = None) -> dict:
    """One streamed turn, measured the way npu-chat experiences it."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
        **(extra or {}),
    }
    out, reasoning = [], 0
    first = first_any = None
    usage = None
    finish = None
    t0 = time.perf_counter()
    with post(f"{base}/v1/chat/completions", body) as resp:
        for line in resp:
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            payload = text[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                thought = (delta.get("reasoning") or "") + (delta.get("reasoning_content") or "")
                reasoning += len(thought)
                piece = delta.get("content")
                if (thought or piece) and first_any is None:
                    first_any = time.perf_counter() - t0
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    out.append(piece)
                finish = choice.get("finish_reason") or finish
    total = time.perf_counter() - t0
    tokens = (usage or {}).get("completion_tokens")
    tps = None
    # completion_tokens counts reasoning too, so time it from the first token
    # of either kind, not the first visible one.
    if tokens and first_any is not None and total > first_any:
        tps = tokens / (total - first_any)
    return {"reply": "".join(out).strip(), "ttft": first, "total": total,
            "tokens": tokens, "tps": tps, "reasoning_chars": reasoning,
            "finish": finish}


def tool_probe(base: str, model: str, extra: dict | None = None) -> tuple[bool, str]:
    body = {
        **(extra or {}),
        "model": model,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "tools": TOOLS,
        "max_tokens": 512,
    }
    try:
        with post(f"{base}/v1/chat/completions", body) as resp:
            d = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read()[:200]!r}"
    choice = d["choices"][0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    if choice.get("finish_reason") == "tool_calls" and calls:
        fn = calls[0].get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return False, f"unparseable arguments {fn.get('arguments')!r}"
        if fn.get("name") == "get_weather" and args.get("city"):
            return True, f"get_weather({json.dumps(args)})"
        return False, f"wrong call {fn.get('name')}({fn.get('arguments')})"
    content = (msg.get("content") or "").strip().replace("\n", " ")
    return False, (f"no tool call, finish_reason={choice.get('finish_reason')}, "
                   f"content={content[:120]!r}")


def fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.fmean(xs):.2f}s +/- {sd:.2f}"


def bench_model(base: str, model: str, runs: int, apples: int, system: str,
                extra: dict | None = None) -> None:
    print(f"== {model} @ {base}")
    # Load and warm; its timing is reported but kept out of the warm stats.
    try:
        cold = stream_once(base, model, system, PROMPTS["arith"], extra)
    except Exception as exc:
        print(f"  FAILED to load/answer: {exc}")
        return
    print(f"  cold: total {cold['total']:.2f}s")

    for name, prompt in PROMPTS.items():
        rs = [stream_once(base, model, system, prompt, extra) for _ in range(runs)]
        ttft = [r["ttft"] for r in rs if r["ttft"] is not None]
        tps = [r["tps"] for r in rs if r["tps"]]
        empty = sum(1 for r in rs if not r["reply"])
        think = max(r["reasoning_chars"] for r in rs)
        speed = f"{statistics.fmean(tps):.1f} tok/s" if tps else "tok/s n/a"
        print(f"  [{name:7}] total {fmt([r['total'] for r in rs])}  "
              f"ttft {fmt(ttft)}  {speed}  reasoning_chars<={think}  "
              f"empty={empty}/{runs}  finish={rs[-1]['finish']}")
        print(f"            reply: {rs[-1]['reply'][:140]!r}")

    right = 0
    for i in range(apples):
        r = stream_once(base, model, system, APPLES, extra)
        ok = bool(APPLES_RIGHT.search(r["reply"]))
        right += ok
        print(f"  [apples {i + 1}] {'ok ' if ok else 'BAD'} {r['reply'][:100]!r}")
    print(f"  apples: {right}/{apples}")

    ok, detail = tool_probe(base, model, extra)
    print(f"  tools: {'PASS' if ok else 'FAIL'} - {detail}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--model", action="append", required=True,
                    help="model to bench (repeat for several)")
    ap.add_argument("--runs", type=int, default=3, help="warm runs per prompt")
    ap.add_argument("--apples", type=int, default=5, help="word-problem repeats")
    ap.add_argument("--no-think", action="store_true",
                    help='send reasoning_effort="none" (ollama honours it; think:false '
                         "and chat_template_kwargs are ignored on its /v1 endpoint)")
    args = ap.parse_args()
    if args.runs < 1 or args.apples < 0:
        ap.error("--runs must be >= 1 and --apples >= 0")

    base = args.url.rstrip("/")
    system = chat_system_prompt()
    for model in args.model:
        bench_model(base, model, args.runs, args.apples, system,
                    {"reasoning_effort": "none"} if args.no_think else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
