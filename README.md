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
Conversation history is kept across turns, so follow-ups work.

One server serves both jobs. `flm serve --asr 1` answers
`/v1/audio/transcriptions` *and* `/v1/chat/completions`, loading the LLM on
first use (~2.3 s) with no restart; the NPU lock serialises the two.

### Model choice

Measured on this box, same prompt, warm:

| model | latency | result |
|---|---|---|
| **`gemma3:1b`** (default) | **1.37 s**, 39.4 tok/s | correct |
| `qwen3:4b` | 9.94 s, 19.4 tok/s | factually wrong |

Asked for the capital of Ohio and one thing it is known for, `qwen3:4b` invented
"the Columbus Monument, a historic landmark featuring a massive statue of a
woman". `gemma3:1b` is both faster and more accurate here, so it is the default.
Replies stream, so first token lands in ~0.5–0.7 s.

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