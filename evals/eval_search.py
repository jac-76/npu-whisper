"""Score npu-chat's web search on current-fact questions.

Answers were verified by hand against 2+ sources on 2026-09-29; they will age.

Runs the real chat_turn() from npu-chat against the desktop model, fresh history
per run. PASS = the verified answer appears in the reply. Also records whether it
searched at all, and wrong claims are visible in the printed replies.

usage: NPU_CHAT_LLM_URL=http://HOST:11434 [PACE=15] evals/eval_search.py LABEL [RUNS]
       PACE = seconds between questions; unpaced runs get engines rate-limited.
"""
import json, os, re, sys, time, types

CONFIG = sys.argv[1]
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
m = types.ModuleType("c")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(compile(open(os.path.join(HERE, "..", "npu-chat")).read(), "c", "exec"), m.__dict__)
m.LLM_URL = os.environ["NPU_CHAT_LLM_URL"].rstrip("/")   # an ollama server
MODEL = os.environ.get("NPU_CHAT_MODEL", "qwen3.5:9b-16k")

QA = [
    ("Who won the most recent Super Bowl?", r"seahawks"),
    ("Who won the World Cup this year?", r"spain"),
    ("Who won the NBA championship this year?", r"knicks"),
    ("Who won the Masters this year?", r"mcilroy"),
    ("Who is the chair of the Federal Reserve?", r"warsh"),
    ("Who won the Stanley Cup this year?", r"hurricanes"),
    ("Which horse won the Kentucky Derby this year?", r"golden tempo"),
    ("Who did Ohio State play in their first football game this season?", r"ball state"),
]

if os.environ.get("ONLY"):
    QA = [x for x in QA if re.search(os.environ["ONLY"], x[1])]

if CONFIG == "think":
    orig = m.request_body
    def rb(model, history, server):
        b = orig(model, history, server); b.pop("reasoning_effort", None); b["max_tokens"] = 4096
        return b
    m.request_body = rb

sysmsg = {"role": "system", "content": m.SYSTEM_PROMPT + m.TOOLS_HINT.format(
    today=time.strftime("%A %B %-d, %Y"))}
sink = open(os.devnull, "w")
passes = searched = 0
src_ok = src_wrong = warn_on_correct = warn_on_wrong = 0
lat = []
for q, key in QA:
    row = []
    for _ in range(RUNS):
        time.sleep(float(os.environ.get("PACE", "0")))   # be a person, not a crawler
        hist = [dict(sysmsg)]
        t0 = time.time()
        stdout, sys.stdout = sys.stdout, sink       # stream_reply prints tokens
        try:
            found = []
            reply, ttft, used, _ = m.chat_turn(MODEL, hist, q, "ollama", True,
                                               log=lambda *a: None, found=found)
        finally:
            sys.stdout = stdout
        dt = time.time() - t0
        lat.append(dt)
        ok = bool(re.search(key, reply, re.I))
        s = any(u["name"] == "web_search" for u in used)
        passes += ok; searched += s
        src = m.attribute(reply, found, q) if (s and found and reply) else None
        if src is not None:
            # a source is right when it actually contains the verified answer
            if re.search(key, src["title"] + " " + src["content"], re.I):
                src_ok += 1
            else:
                src_wrong += 1
        elif s and found and reply:
            if ok:
                warn_on_correct += 1
            else:
                warn_on_wrong += 1
        row.append(f"{'PASS' if ok else 'FAIL'}{'' if s else '(no search)'} {dt:.1f}s")
        if not ok:
            qs = [u["args"].get("query") for u in used]
            print(f"    FAIL queries={qs} reply: {' '.join(reply.split())[:110]!r}")
    print(f"{q[:52]:52} {' | '.join(row)}")
n = len(QA) * RUNS
lat.sort()
print(f"\n{CONFIG}: {passes}/{n} correct, searched in {searched}/{n}, "
      f"median {lat[len(lat)//2]:.1f}s, max {lat[-1]:.1f}s")
print(f"sources: {src_ok} cite a result containing the answer, {src_wrong} cite one "
      f"that does not; 'not in results' warnings: {warn_on_wrong} on wrong answers, "
      f"{warn_on_correct} on correct ones")
