# npu-whisper

**Whisper ASR on the AMD Ryzen AI NPU**, three ways: one-shot file
transcription (`whisper-npu`), push-to-talk dictation into any window
(`npu-dictate`), and a voice chat whose LLM also runs on the NPU (`npu-chat`).
All of them drive a [FastFlowLM](https://fastflowlm.com/) server over its
OpenAI-compatible API, started on demand. Everything runs locally; nothing
leaves the machine.

Verified on a **MSI Stealth A16 AI+** (AMD Ryzen AI 9 365, XDNA2 NPU, 8 columns)
under Omarchy/Arch Linux.

## Why the NPU?

Running Whisper on the NPU keeps the CPU and GPU free and uses far less power than
CPU transcription. On this hardware a 28.5 s clip transcribes in **~5.4 s
(RTF ≈ 0.19)** with the LLM-sized `whisper-large-v3-turbo` model.

## Requirements (driver stack)

The NPU must be usable via XRT before this tool can do anything:

- kernel `>= 6.14` with the `amdxdna` driver (or the out-of-tree
  [xdna-driver](https://github.com/amd/xdna-driver))
- `xrt` + `xrt-plugin-amdxdna` — confirm with `xrt-smi examine`
- `fastflowlm` — NPU runtime; `flm validate` should pass
- the `whisper-v3:turbo` model pulled via `flm pull whisper-v3:turbo`
- unlimited memlock for the user (see `flm validate` → "Memlock Limit")

## Install

```sh
install -m755 whisper-npu ~/.local/bin/
```

## Usage

```sh
whisper-npu file.ogg            # transcribe one file
whisper-npu a.ogg b.wav -j      # multi-file, full JSON per file
whisper-npu --status            # is the ASR server up?
whisper-npu --stop              # unload the model / free the NPU
```

The first call starts the FastFlowLM server (~10–15 s model load) and leaves it
warm; subsequent calls return in seconds. Supported audio: anything FFmpeg
decodes (wav, mp3, ogg, m4a, flac, …).

## Benchmark

Reproducible timing is built in — `bench.py` times real `whisper-npu`
invocations, pulls the audio length from the file itself (ffprobe), and
reports mean ± stddev plus real-time factor for cold and warm runs:

```sh
python3 bench.py --runs 10 --warmup 2 sample.ogg
python3 bench.py --runs 10 --warmup 2 --power sample.ogg   # + RAPL power/energy
```

Verified on this machine (Ryzen AI 9 365 / XDNA2, 30.0 s clip, 10 runs with the
first 2 discarded as warm-up):

| Metric | Value |
|---|---|
| Warm mean | **5.2 s** (σ 0.04 s) |
| **RTF** | **≈ 0.17–0.19** (~5–6× faster than real time) |
| CPU-core power while running | **~0.8 W** (idle ~0.4 W) |
| Energy per transcription | **~45 J** over idle |

System load sampled during those runs: CPU 4.3% mean against a 2.2% idle
baseline, integrated GPU 10% against 7% idle (that delta is desktop
compositing), discrete GPU flat at 0%. The transcription work doesn't show up on
any of them — it's on the NPU.

**CPU baseline:** `bench_cpu.py` runs the *same clip and model* through
`whisper.cpp` on the CPU under the same RAPL measurement. On this box (16
threads): ~6.8 s, **~400 J per transcription over idle** — roughly **10× the
NPU's energy** for a ~25 % slower result. See `bench_cpu.py --help` for setup
(build whisper.cpp from source; the Arch package's ggml backend is broken).

## Voice chat on the NPU (`npu-chat`)

`npu-chat` is a terminal voice assistant where **both halves run on the NPU**:
speech recognition with `whisper-large-v3-turbo` and generation with an LLM, over
the same FastFlowLM server. **Push to talk is Enter** — press Enter to start
recording, Enter again to stop. Typing a line sends it as text instead.

```sh
install -m755 npu-chat ~/.local/bin/
npu-chat                             # NPU ASR + NPU LLM + spoken reply
npu-chat --model qwen3:4b            # pick another LLM
npu-chat --voice en_US-amy-medium    # pick another voice
npu-chat --no-speak                  # text only
```

```
you ● recording - press Enter to stop
you What is 17 times 3?
npu 17 times 3 is 51.
[gemma3:1b - first token 0.66s, total 0.96s]
```

Commands: `/model NAME`, `/models`, `/voice NAME`, `/voices`, `/speak`,
`/reset`, `/help`, `/quit`.
Conversation history is kept across turns, so follow-ups work — and FastFlowLM
reuses the prefix, reporting e.g. `Matched 8 out of 10 messages (2 new to
prefill)`, so a long conversation does not re-cost the whole history each turn.

The system prompt states what the assistant actually is — model id, that it runs
on the NPU under FastFlowLM, the ASR model, and the piper voice — and is rebuilt
when `/model`, `/voice` or `/speak` changes it. A 1B model cannot know any of
that on its own, and "what model are you?" is a natural thing to ask by voice.
It still garbles the supplied facts sometimes; it is a 1B model.

One server serves both jobs. `flm serve --asr 1` answers
`/v1/audio/transcriptions` *and* `/v1/chat/completions`, loading the LLM on
first use (~2.3 s) with no restart; the NPU lock serialises the two.

### Model choice

Default is **`qwen3.5:4b`**, chosen on measurement. Three FastFlowLM models, same
prompts, warm, 3 runs each:

| model | factual | arithmetic | two-step word problem | tok/s |
|---|---|---|---|---|
| **`qwen3.5:4b`** (default) | 2.59 s ✅ | 1.28 s ✅ | 2.18 s **✅ 5/5** | 14.2 |
| `gemma3:1b` | 1.32 s ✅ | 0.79 s ✅ | 0.75 s **❌ 0/5** | 39.6 |
| `qwen3:4b` | 8.93 s ✅ | 8.33 s ❌ leaks `<think>` | 7.18 s ✅ | 19.3 |

The word problem is *"I have 5 apples, eat 2, then buy 4 more. How many do I
have?"* (answer 7). `gemma3:1b` answered **"You have 3 apples left"** five times
out of five — it drops the final clause every time. `qwen3.5:4b` got it right
five times out of five. Being 2× slower per reply and still under three seconds
is the better trade for something you talk to.

`qwen3:4b` is worth avoiding: it leaks raw `<think>` blocks into `content` and
burned a 200-token budget mid-thought on `17 × 3`.

### Markdown is stripped before speaking

Models emit markdown even when told not to, and piper pronounces it:
`"You now have **7** apples."` is spoken as *"You now have asterisk-sterisk-seven
asterisk-ster apples"* — 4.23 s of audio instead of 1.96 s. `for_speech()`
removes bold, italics, code, headings, bullets, quotes and link syntax before
synthesis, while leaving `snake_case` identifiers alone. Printed text and the
transcript keep the model's raw output.

### Using an LLM the NPU cannot run (`--llm-url`)

FastFlowLM only runs models it ships an NPU2 build for — 38 of them, and
**Granite is not among them** (`flm pull granite4.2:3b` → "Model not found").
For agent work, **tool calling on FastFlowLM depends on the model.** Given the
same `tools` array on FLM 1.0.4, `qwen3.5:4b` returns a real call
(`finish_reason: tool_calls`, `get_weather({"city": "Columbus, Ohio"})`), while
`gemma3:1b` ignores it (`finish_reason: stop`) and has been seen to *pretend*
it called the tool and fabricate the result. An earlier version of this README
said FastFlowLM has no tool calling at all, which was wrong: only `gemma3:1b` had
been tested.

So the LLM endpoint is separable. ASR stays on the NPU; point `--llm-url` at any
OpenAI-compatible server:

```sh
# terminal 1 - granite on CPU + GPU, with real tool calling
llama-server -m granite-4.2-3b-Q4_K_M.gguf --jinja --reasoning off   --host 127.0.0.1 --port 8080 -c 8192 -ngl 99

# terminal 2 - speech on the NPU, generation from granite
npu-chat --llm-url http://127.0.0.1:8080 --model granite
```

`--reasoning off` matters: granite 4.2 is a reasoning model and llama.cpp
preserves thinking by default, which spent an entire 200-token budget on
`reasoning_content` and returned `finish_reason: length` with empty `content`.
With it off the answer comes straight back, and tool calls got faster too
(1.39 s → 0.40 s).

Measured on this box, same prompt, warm:

| model | where | latency | tok/s | tool calls |
|---|---|---|---|---|
| `gemma3:1b` | NPU | 1.20 s | 40.2 | **no** (silently ignored) |
| `granite-4.2-3b` Q4_K_M | CPU + RTX 4070 | **0.22 s** | **94.3** | **yes** (`finish_reason=tool_calls`) |

The system prompt tracks this honestly — with `--llm-url` set, the assistant is
told it is *not* on the NPU, and where it is (see `--llm-where` below), so asking it where it runs
still gets a true answer.

### A bigger LLM on another machine's GPU (ollama)

Speech stays on the laptop (NPU whisper, piper voice, the laptop's mic and
speakers); only generation goes over the network to a desktop with an **RTX
3060 Ti (8 GB)** running ollama:

```sh
export NPU_CHAT_LLM_URL=http://192.168.1.36:11434
export NPU_CHAT_LLM_WHERE="the downstairs desktop's NVIDIA RTX 3060 Ti"
export NPU_CHAT_MODEL=qwen3.5:9b-16k
npu-chat
```

`--llm-where` is what the model is told about where it runs. Without it a
remote server is described as "a separate machine at HOST", because nothing on
the wire says what hardware is behind it. If the server is unreachable,
npu-chat says so at startup and exits. It does **not** fall back to the NPU,
since a reply from a model you did not pick would be a surprise.

**Context is the constraint on 8 GB, not weights.** A model's context is set
per model in ollama (`FROM qwen3.5:9b` + `PARAMETER num_ctx 16384` in a
Modelfile, then `ollama create qwen3.5:9b-16k`). Clients cannot pass it through
`/v1`. Measured placement (`ollama ps`):

| model | context | size | placement |
|---|---|---|---|
| `qwen3.5:9b` | 16k | 5.9 GB | 100% GPU |
| `qwen3.5:9b` | 32k | 7.2 GB | 18% CPU / 82% GPU |
| `qwen3.5:4b` | 64k | 5.3 GB | 100% GPU |
| `qwen3:8b` | 8k | 6.2 GB | 100% GPU |
| `qwen3:8b` | 16k | 7.8 GB | 15% CPU / 85% GPU |
| `granite4.2:3b` | 32k | 5.0 GB | 100% GPU |
| `llama3.1:8b` | 32k | 9.5 GB | 32% CPU / 68% GPU |

**Thinking must be off.** Qwen3.5 on ollama reasons by default and spent the
whole 512-token budget on it: empty reply, `finish_reason: length`, on every
prompt. npu-chat sends `reasoning_effort: "none"` to ollama servers, which
brought reasoning to zero characters. `"think": false` and
`chat_template_kwargs` are silently ignored on ollama's `/v1` endpoint.

Measured from the laptop over WiFi with `bench_llm.py --no-think`, warm, 3 runs
per prompt, word problem ×5:

| model | where | factual | first token | tok/s | word problem | tool call |
|---|---|---|---|---|---|---|
| `qwen3.5:4b` | laptop NPU (FLM) | 3.24 s | 1.11 s | 15.5 | 5/5 | yes |
| **`qwen3.5:9b-16k`** | desktop 3060 Ti | 0.80 s | 0.07 s | 60.5 | **5/5** | yes |
| `qwen3.5:4b-32k` | desktop 3060 Ti | 0.57 s | 0.06 s | 87.5 | 4/5 | yes |
| `qwen3:8b-8k` | desktop 3060 Ti | 0.42 s | 0.04 s | 76.6 | 5/5 | yes |
| `granite4.2:3b-32k` | desktop 3060 Ti | 0.23 s | 0.04 s | 126.2 | 5/5 | yes |

`qwen3.5:9b-16k` is the pick for talking: 5/5, and the one whose Ohio answer
held up (`qwen3.5:4b` claimed Columbus hosts "America's first National Mall").
Only one model fits in 8 GB at a time, so switching models costs a cold load.
npu-chat hides it by asking ollama to load the model in the background at
startup (and on `/model`). From an unloaded model, the first turn sent 6 s after
startup got its first token in 0.14 s. Without the pre-load the same cold turn
took 3.3 s.

```sh
python3 bench_llm.py --no-think --url http://192.168.1.36:11434 \
  --model qwen3.5:9b-16k --model granite4.2:3b-32k
```

### Web search and grow status (tools)

With an ollama model, npu-chat gives the model two tools. It asks whether the
server is ollama, because that is the only server it was measured on.

- **`web_search`** goes to a self-hosted [SearXNG](https://github.com/searxng/searxng)
  on the laptop (`NPU_CHAT_SEARX_URL`, default `http://127.0.0.1:8888`). The
  top 8 results go back to the model, plus SearXNG's direct answers when
  there are any. **Searching sends your question to the public search engines**,
  from your IP but not tied to any account. That is the one thing in npu-chat
  that leaves the LAN, and a `· searching: …` line shows every time it happens.
- **`grow_status`** reads what the grow daemons have already cached
  (`acinf bar`, `vivosun bar`, `yinmik bar`, 0.05 s). It never polls the cloud
  APIs, because AC Infinity answers extra polling with `403`.

`--no-tools` turns both off. You hear "Let me check." while it looks things up.

**Don't leave it to the model to decide when to look.** On a set of 8 current
facts, each answer checked by hand against at least two sources (the 2026 Super
Bowl, World Cup, NBA title, Masters, Fed chair, Stanley Cup, Kentucky Derby and
Ohio State's opener), 3 fresh runs each, `qwen3.5:9b-16k`:

| version | correct | searched |
|---|---|---|
| model decides when to search | 13/24 | 21/24 |
| + year in the tool description, more results | 13/24 | 15/24 |
| **npu-chat searches first when the question is about the present** | **21/24** | 24/24 |

When it decided alone, the model answered "Jerome Powell" three times out of three
without searching. It searched "who won the Masters this year **2024**" in 2026.
And after finding "Seahawks 29–13" it called that "an alternate timeline" and gave
its training-era answer instead. So a question with words like *who won*,
*this year*, *latest*, *current* or *price of* is now searched by npu-chat
itself, with the year added, and the results are attached to your message. The
model can still search again on its own. Grow questions work the same way:
left to the model, 1 of 3 runs said "I can't see real-time grow room
readings". With the readings attached whenever you mention the tent,
reservoir, pH, humidity and so on, it got 6 of 6.

The three remaining misses were all the Fed chair. At that moment the search
engines returned furniture ("chairs") and Powell's official bio, not the news
of his successor. Engines don't return the same results twice: a few minutes
later the same query did find it. Median time to answer was 2.2 s, search
included.

**Engines block heavy use.** About 150 searches in 30 minutes got Brave and
Google CSE rate-limited and DuckDuckGo blocked, which left zero results. With
nothing to go on, the model said it "couldn't find" things rather than
inventing them. SearXNG now also queries Bing, Google and Qwant, and when every
engine is refusing, the tool result says *search is unavailable*, so the model
says that instead of claiming nothing exists.

Any advice the model adds on top of the grow readings is its own opinion, not a
rule anyone gave it. Asked the same question, it called pH 5.8 "a bit low" in one
run and "perfect" in another.

SearXNG setup used here, local only:

```sh
mkdir -p ~/.config/searxng && openssl rand -hex 32 > ~/.config/searxng/secret
# settings.yml: use_default_settings: true; server.limiter: false;
# search.formats: [html, json]; enable bing, google, qwant under engines:
docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 \
  -v ~/.config/searxng/settings.yml:/etc/searxng/settings.yml:ro \
  -e SEARXNG_BASE_URL=http://127.0.0.1:8888/ \
  -e "SEARXNG_SECRET=$(cat ~/.config/searxng/secret)" searxng/searxng:latest
```

Mount the settings file read-only. With the directory mounted read-write, the
container changed the file's owner to its own user.

### Grow control, confirmed before anything changes

A third tool, **`grow_control`**, changes one output: VIVOSUN light
(0 or 25–100 %), duct fan (0–10) or circulation fan (0–10 or `natural`), or an
AC Infinity port 1–4 (speed 0–10, or a mode: on, off, auto, vpd, cycle,
schedule). It runs the same `vivosun set` / `acinf set|mode` commands as the
control windows.

**The request is parsed by npu-chat, not left to the model.** Offered the tool,
`qwen3.5:9b` called it for only 2 of 12 spoken requests, and 3 of 12 with
`tool_choice` forcing it, since ollama doesn't enforce that
(`evals/eval_control.py`). It never called it for the light or the duct fan.
Once it said "I'll increase your circulation fan to speed 5 right now" and
called nothing. The vocabulary is small (light, duct fan, circulation fan,
port 1–4, a number, off, natural, a mode), so "turn the circulation fan up to
5", "turn the grow light off", "switch port 4 to schedule" and "turn port two
off" are parsed directly and never reach the model. Anything relative ("up by
2") or ambiguous ("turn up the fans") goes to the model. If a request to change
something ends with nothing having run, you hear **"Nothing was changed."**,
whatever the model said.

Parsed or not, a request is never run as it comes:

1. npu-chat checks it against that list and refuses anything else, such as light
   at 20 %, port 7, or `5; rm -rf ~`. The model hears "not allowed", and nothing
   runs.
2. It builds the question from the cached readings, **including the side effect
   on mode**, then shows it and speaks it:

   ```
   ? AC Infinity port 4: set it to 6 out of 10? It is in schedule mode at 7 out
     of 10 now. That ends its schedule mode and makes it manual.
   Enter do it · x cancel
   ```

   Both CLIs change modes when they set a level. `acinf set` puts the port in
   manual "on" (0 means off), and `vivosun set` puts that output in manual. A
   voice command that silently ended a schedule would be the wrong kind of
   surprise.
3. Only Enter, `y` or `yes` runs it. Anything else, or no answer, cancels, and
   the model is told nothing changed. The command's own output (`vivosun`
   reports confirmed / already / unconfirmed) goes back to the model and into
   the session log.

Tested live on the real hardware with values it already had: `circ 3`, and
VIVOSUN answered "already at 3/10 in manual mode; nothing changed". AC
Infinity port 1 at 3: the write went through and the port read back unchanged.
Two real requests answered with `x` changed nothing.

The confirmation is part of the interactive loop only. Nothing else that calls
`chat_turn()`, such as the evals and tests, can change hardware.

### Where an answer came from

After a search, npu-chat checks the reply against the results itself instead of
trusting the model to cite. The supporting result is the one that shares the
most of the reply's names and numbers ("Seahawks", "Patriots", "29"). Its site
is printed with the link, and spoken if the model did not already name it
("Source: FIFA."). If **no** result supports the reply, you hear "I could not
find that in the search results, so treat it with care." An earlier run
invented "Chiefs 61–9, as reported by ESPN" from results that said nothing of
the kind. That reply matches none of them, so it would have been flagged.

### Picking up where you left off (`--resume`)

`npu-chat --resume` reloads the user/assistant turns of the most recent chat
session from its transcript and keeps writing to the same file. Discarded and
failed turns are skipped, and so are agent-mode sessions. Search results and
grow readings are not restored, only what was said.

### Talking to a coding agent (`--harness claude`)

The same push-to-talk loop can drive **Claude Code** instead of a chat model.
You speak a request, whisper on the NPU transcribes it, a local model on the
desktop GPU does the work through Claude Code's tools, and piper reads out a
short summary. Nothing leaves the LAN.

```sh
export NPU_CHAT_LLM_URL=http://192.168.1.36:11434      # ollama
npu-chat --harness claude --workdir ~/Dev/some-repo \
  --agent-allow "Bash(.venv/bin/python -m pytest:*)"
```

A real session (typed turns, some tool lines omitted, first reply cut):

```
you Why do the tests fail? Run them to find out, but do not change anything yet.
· Bash .venv/bin/python -m pytest -v 2>&1
  ↳ failed: Exit code 1 ...
· Read tally/stats.py
agent I found the bug. In `tally/stats.py` line 11, the formula for generating windows is incorrect: …
[qwen3.5:4b-64k - 7 tool calls, 19.2s]

you Now fix the bug in tally/ and run the tests to confirm. Do not change tests/.
· Edit tally/stats.py
· Bash .venv/bin/python -m pytest -v
agent I fixed the bug in `tally/stats.py` by changing `range(len(xs) - n)` to `range(len(xs) - n + 1)`. The original code excluded the last valid window from generating averages. All three tests now pass.
[qwen3.5:4b-64k - 2 tool calls, 6.0s]
 tally/stats.py | 2 +-
```

It does not always follow instructions. In an earlier run it made the fix during
the "don't change anything yet" turn.

The default model is `qwen3.5:4b-64k` (`NPU_CHAT_HARNESS_MODEL`). **It needs a 64k
context.** At 32k, Claude Code's own prompt plus a few tool results refilled the
window within three turns and autocompact thrashed in 9 of 16 runs. At 64k it
thrashed in none of 12. In a small benchmark on a four-file repo it passed 9/9
tasks (read, fix, extend). That says little about large real repositories.

Claude Code runs as **one long-lived process** speaking stream-json, so turns
share context. **Ctrl+C** stops the current turn, and the next turn resumes the
same session (`--resume`), with history intact.

#### What the agent may do

Nothing can be approved by voice mid-turn, so anything not explicitly allowed
is refused (`--permission-mode dontAsk`). Each rule below exists because a
looser one was tried and leaked:

- **`--workdir` must be a git repository**, and `$HOME` or `/` is refused. Every
  edit shows up in `git diff`; the loop prints `git diff --stat` after any turn
  that edits, and `/diff` shows the full diff.
- **File access is scoped to the workdir**: `Read(./**)`, `Edit(./**)`. A plain
  `Read` rule let the model read `/etc/hostname` and `~/.bashrc`. The scoped
  rules refused both.
- **Git rules are exact** (`git status`, `git diff`, `git diff --stat`). A
  `git diff:*` prefix rule would allow `git diff --no-index` on any file.
- Claude Code also lets some read-only commands (`ls`, `find .`,
  `git diff <path>`) run inside the workdir without a rule. `ls /etc`, `ls ~`
  and `git diff … /etc/hostname` were all refused.
- **`--agent-allow` adds rules**, typically a test runner. Relative commands are
  also allowed by their absolute path, because models call `.venv/bin/python`
  as `/full/path/.venv/bin/python`. The agent is told which commands it may run.
  Before that, it guessed `python -m pytest` and was refused ten times in one
  turn. **Allowing a test runner allows running code the agent can edit**, so
  treat it as that.
- `--bare` keeps your own Claude Code hooks, plugins, MCP servers and memory out
  of the agent. The env vars pointing Claude Code at ollama are set on the
  child process only.

#### Every spoken request is confirmed first

whisper invents text from room noise. On the first test of this gate, 3 s of
silence came back as `*Pewing* *Pewing* *Pewing`. An agent would act on that,
so a voice transcript is shown with `Enter send · x discard · or type a
correction`. Discarded transcripts are logged with `"sent": false`. Typed turns
skip the prompt.

Replies are printed in full, but only the first three sentences, stripped of
markdown, are spoken. A failed model load (the server answers `API Error: …`)
is announced as an error and never read out as an answer.

### Session transcripts

Every session is written to `~/.local/state/npu-chat/<timestamp>.{md,jsonl}` —
the Markdown to read, the JSONL for tooling, both with per-turn `ttft_secs`,
`total_secs`, `audio_secs` and the model used. `--no-transcript` turns it off.

This exists because nothing else kept the conversation: FastFlowLM logs only the
final stream chunk, whose `content` is `null`, so a 12-turn session left twelve
log lines and zero recoverable replies.

### Dead-air trimming

Enter-to-start/Enter-to-stop means recording runs while you think, so most of a
capture is silence. Captures are trimmed with ffmpeg `silenceremove` before
transcription — measured live, 6.9 s → 3.9 s.

The threshold is `NPU_CHAT_TRIM_DB` (default `-30dB`). It matters that this was
checked rather than assumed, because this microphone's noise floor is loud: room
noise measures about **-19.5 dBFS** against speech at **-18.3**, barely 1.2 dB
apart. On a padded sample (4 s noise + 1.1 s speech + 4 s noise) the transcript
came back identical at every threshold tried:

| threshold | 9.05 s becomes | transcript |
|---|---|---|
| `-40dB` | 8.95 s | `How are you today?` |
| `-30dB` (default) | 6.77 s | `How are you today?` |
| `-25dB` | 3.08 s | `How are you today?` |
| `-20dB` | 2.68 s | `How are you today?` |

`-30dB` is the conservative default; set `-25dB` for a much bigger cut. A trim
that would leave less than `NPU_CHAT_MIN_SECS` of audio is discarded and the
untrimmed clip is used.

### Spoken replies: piper, on the CPU

Replies are spoken by default with [piper](https://github.com/rhasspy/piper)
(`piper-tts`), voice **`en_GB-alan-medium`**. Voices are looked up in
`~/.local/share/piper` then `~/.local/share/piper-voices`; `/voices` lists what
is installed and `/voice NAME` switches at runtime. `--no-speak` turns it off.

**Speech starts on the first sentence, not after the whole reply.** Text is cut
into sentences as it streams, one thread synthesises and another plays, so
sentence two is synthesised while sentence one is heard. piper runs in-process
with the voice loaded once. A `piper-tts` process per call spent 0.75 s
reloading the model every time, while in-process synthesis of the same sentence
took 0.06 s. Fenced code is never read aloud.

Time from sending a turn to the first audio, desktop `qwen3.5:9b-16k`, warm:

| reply | before (whole reply, then a piper process) | now, live |
|---|---|---|
| "What is 17 times 3?" | 1.19 s | **0.36 s** |
| Ohio, one sentence | 1.72 s | **0.58 s** |
| heat pump, several sentences | 2.36 s | **0.64 s** (reply took 2.3 s) |

Each turn logs `first_audio_secs` in the session JSONL.

**Barge-in:** press Enter (or type anything) while it is talking and playback
stops at once, queued sentences are dropped, and your next recording starts.
Measured live: 0.4 s after a new question, the old answer was silent and the
new one was playing.

TTS is the one part that is **not** on the NPU — FastFlowLM offers only `--asr`
and `--embed`, no synthesis — so piper runs on the CPU. Recognition and
generation stay on the NPU.

One trap worth repeating: **never pass `--output-raw` to `piper-tts`.** It
redirects audio to stdout and leaves `-f` empty, so playback gets silence.

### Speech gate: Silero VAD ahead of whisper

whisper large-v3-turbo invents text from room noise, so every capture is
checked for speech first ([Silero VAD](https://github.com/snakers4/silero-vad)
v6.2.3, ONNX, on the CPU). A capture with no speech is discarded before it
reaches whisper or the model. Install the model once:

```sh
mkdir -p ~/.local/share/silero-vad
curl -L -o ~/.local/share/silero-vad/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx
```

Without it (or without `numpy`/`onnxruntime`) npu-chat says the gate is off and
behaves as before. `NPU_CHAT_VAD=0` turns it off deliberately.

**The threshold was measured on this microphone, and the usual one is wrong
here.** Peak speech probability per capture:

| capture | peak probability | Silero's usual 0.5 | whisper said |
|---|---|---|---|
| room noise ×6 | 0.011 – 0.074 | rejected | "Thank you.", "E aí", "Продолжение следует..." |
| quiet speech ×3 | 0.163 – 0.414 | **rejected** | the right words (2 of 3 exact) |
| normal speech ×3 | 0.954 – 0.997 | kept | the right words |

At 0.5, quiet but perfectly intelligible speech would be thrown away. So a
capture is kept if any 32 ms frame reaches **0.12** (`NPU_CHAT_VAD_MIN`),
between the two groups. The "speech" here is piper played through the laptop
speakers and heard by the mic, not a person, and there are few samples. Each
capture's `vad_max_p` is logged, including discards (`"via": "vad-discarded"`),
so the threshold can be re-checked against real use.

`--confirm` adds the agent mode's prompt to chat mode: see the transcript,
Enter to send, `x` to discard, or type a correction.

## Push-to-talk dictation (`npu-dictate`)

`npu-dictate` turns the same NPU server into hold-to-talk dictation: it records
while a key is held, transcribes on the NPU, and types the result into whatever
window has focus.

```sh
install -m755 npu-dictate ~/.local/bin/

npu-dictate start      # begin recording (bind to key press)
npu-dictate stop       # stop, transcribe on the NPU, type the text
npu-dictate toggle     # start if idle, stop if recording
npu-dictate cancel     # discard without typing
npu-dictate status     # idle | recording | transcribing
```

Bind press and release to one key. Under Omarchy/Hyprland, in
`~/.config/hypr/bindings.lua`:

```lua
o.bind("F10", "Start NPU dictation (push-to-talk)", "npu-dictate start")
o.bind("F10", "Stop NPU dictation (push-to-talk)", "npu-dictate stop", { release = true })
```

Pipeline: `pw-record` (16 kHz mono) → `ffmpeg` normalise → NPU transcription →
`wtype`, falling back to `wl-copy` if typing fails. It starts the ASR server on
demand via `whisper-npu --start`. Transcription is on the NPU; recording,
resampling and typing use a little CPU.

### Why there is no LLM cleanup by default

FastFlowLM serves ASR and an LLM from the same port, so piping the transcript
through an NPU LLM for punctuation cleanup is one HTTP call away, and
`--clean` does exactly that. It is **off by default**, because measured on this
box `gemma3:1b` was not worth it — three identical requests, warm:

| | result |
|---|---|
| latency | +1.1 s warm (2.3 s extra on first load) |
| filler words | not removed (`um,` survived every run) |
| capitalization | first word left lowercase |
| run 3 | rewrote **"NPU"** as **"NumPy"** |

`whisper-large-v3-turbo` already returns punctuated, capitalized text, so the
cleanup pass cost latency and risked corrupting exactly the technical
vocabulary dictation needs. `--clean` remains for experimentation.

### Known limitation: hallucination on non-speech

`whisper-large-v3-turbo` invents plausible sentences from room noise. Captures
of non-speech on this box returned `"So, like, this always happens."` and
`"that like if you go from having a"`.

There is no good gate for it, and both candidates were measured and rejected:

- **The server exposes no confidence signal.** `response_format=verbose_json`
  returns `{model, text}` only — no segments, no `no_speech_prob`.
- **Audio energy does not separate speech from noise on this microphone.**
  All three captures below are from the same mic through this pipeline:

  | capture | RMS (full-scale) | peak | transcript |
  |---|---|---|---|
  | speech | 0.112 | 0.913 | correct |
  | noise | 0.104 | 0.881 | hallucinated |
  | noise | 0.060 | 0.271 | hallucinated |

  Real speech sits 8 % above a hallucinating noise capture. Any RMS threshold
  either passes the hallucinations or discards real speech.

So `npu-dictate` asks **`npu-vad`** first: the same Silero speech check and 0.12
cutoff as npu-chat (see "Speech gate"), in about 0.13 s. With no speech in
the capture it types nothing and shows "no speech detected". Tested end to
end with a stand-in `wtype`: a room-noise capture (peak 0.014) was discarded,
while normal speech (0.997) and quiet speech (0.414) were transcribed and typed
correctly. If the check can't run (no model or onnxruntime), dictation carries
on unfiltered, as before. A duration floor (`NPU_DICTATE_MIN_SECS`, 0.4 s) still
applies, and each capture's RMS, peak and speech score go to
`~/.local/state/npu-dictate/dictate.log`.

```sh
install -m755 npu-vad ~/.local/bin/
```

## How it works

```
audio file ──▶ curl POST /v1/audio/transcriptions ──▶ FastFlowLM (flm serve --asr 1)
                                                        │
                                                        ▼
                                              XRT / amdxdna → NPU
                                              whisper-large-v3-turbo (q4nx)
                                                        │
                                                        ▼
                                              transcript text (stdout / --json)
```

The wrapper manages the server lifecycle (`nohup`, pidfile + log under
`~/.local/state/flm/`), waits for readiness against `/v1/models`, and prints
plain transcripts by default — or the full OpenAI-style JSON with `--json`.

Also usable directly as a raw API server:

```sh
flm serve --asr 1
curl http://127.0.0.1:52625/v1/audio/transcriptions -F file=@clip.ogg -F model=whisper-v3
```

## Tests

```sh
python3 -m venv --system-site-packages .venv && ./.venv/bin/pip install pytest
./.venv/bin/python -m pytest
```

(Validates script syntax, CLI behavior, and driver presence — no NPU required.)

## License

MIT