"""Does the model turn a spoken control request into a valid grow_control call?

Nothing is executed: the call is only checked with plan_control(), the same
validation npu-chat applies before asking the user. Compares letting the model
choose against forcing the grow_control tool (tool_choice).

usage: NPU_CHAT_LLM_URL=http://HOST:11434 evals/eval_control.py [RUNS]
"""
import json, os, sys, time, types, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
m = types.ModuleType("c")
exec(compile(open(os.path.join(HERE, "..", "npu-chat")).read(), "c", "exec"), m.__dict__)
URL = os.environ["NPU_CHAT_LLM_URL"].rstrip("/")
MODEL = os.environ.get("NPU_CHAT_MODEL", "qwen3.5:9b-16k")
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# (request, the argv a correct call validates to)
CASES = [
    ("Turn the circulation fan up to 5.", ["vivosun", "set", "circ", "5"]),
    ("Set AC Infinity port 4 to 6.", ["acinf", "set", "4", "6"]),
    ("Turn the grow light off.", ["vivosun", "set", "light", "0"]),
    ("Put the duct fan on 4.", ["vivosun", "set", "duct", "4"]),
]
system = (m.SYSTEM_PROMPT + m.TOOLS_HINT.format(today=time.strftime("%A %B %-d, %Y")))
ACINF = m._cached_tooltip("acinf")
VIVO = m._cached_tooltip("vivosun")


def call(text, force):
    content, _ = m.with_grow_context(text)
    body = {"model": MODEL, "reasoning_effort": "none", "max_tokens": 300,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "tools": m.tools_for()}
    if force:
        body["tool_choice"] = {"type": "function", "function": {"name": "grow_control"}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    msg = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]
    return msg.get("tool_calls") or [], msg.get("content") or ""


parsed = sum(1 for text, want in CASES
             if (pc := m.parse_control(text)) and m.plan_control(*pc, ACINF, VIVO)[0] == want)
print(f"npu-chat's own parser: correct {parsed}/{len(CASES)} (deterministic)\n")

for force in (False, True):
    called = valid = 0
    for text, want in CASES:
        row = []
        for _ in range(RUNS):
            calls, content = call(text, force)
            gc = [c for c in calls if c["function"]["name"] == "grow_control"]
            if not gc:
                row.append("no call")
                continue
            called += 1
            a = json.loads(gc[0]["function"]["arguments"] or "{}")
            try:
                argv, _ = m.plan_control(str(a.get("device")), str(a.get("target")),
                                         str(a.get("value")), ACINF, VIVO)
                ok = argv == want
            except ValueError:
                argv, ok = None, False
            valid += ok
            row.append("OK" if ok else f"BAD {a}")
        print(f"{'forced' if force else 'chosen':6} {text:36} {' | '.join(row)}")
    n = len(CASES) * RUNS
    print(f"{'forced' if force else 'chosen'}: grow_control called {called}/{n}, "
          f"correct arguments {valid}/{n}\n")
