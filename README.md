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
More importantly for agent work, **FastFlowLM has no tool calling**: pass
`tools` and it is silently ignored, and a small model will then *pretend* to
call the tool and fabricate the result.

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
told it runs on the CPU and GPU rather than the NPU, so asking it where it runs
still gets a true answer.

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

Measured here: 859 ms to synthesise 2.65 s of speech (RTF ≈ 0.32) at 22.05 kHz.

TTS is the one part that is **not** on the NPU — FastFlowLM offers only `--asr`
and `--embed`, no synthesis — so piper runs on the CPU. Recognition and
generation stay on the NPU.

One trap worth repeating: **never pass `--output-raw` to `piper-tts`.** It
redirects audio to stdout and leaves `-f` empty, so playback gets silence.

### Same hallucination caveat

`npu-chat` shows you the transcript before it sends anything, so a bad
recognition is visible rather than silent. See the `npu-dictate` section below
for why non-speech audio produces invented sentences and why no gate is
available.

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

So `npu-dictate` enforces only a duration floor (`NPU_DICTATE_MIN_SECS`,
default 0.4 s) and logs RMS/peak per capture to
`~/.local/state/npu-dictate/dictate.log`. Push-to-talk is what actually bounds
the exposure: audio is captured only while the key is held. A real fix needs a
VAD pass (e.g. Silero) ahead of transcription.

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
python3 -m venv .venv && ./.venv/bin/pip install pytest
./.venv/bin/python -m pytest
```

(Validates script syntax, CLI behavior, and driver presence — no NPU required.)

## License

MIT