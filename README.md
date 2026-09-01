# npu-whisper

One-shot **Whisper ASR on the AMD Ryzen AI NPU** — a zero-dependency Bash wrapper
that starts the [FastFlowLM](https://fastflowlm.com/) ASR server on demand and
transcribes audio over its OpenAI-compatible API. Everything runs locally on the
NPU; nothing leaves the machine.

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
```

Verified on this machine (Ryzen AI 9 365 / XDNA2, 30.0 s clip, 10 runs with the
first 2 discarded as warm-up):

| Metric | Value |
|---|---|
| Warm mean | **5.17 s** (σ 0.04 s, n = 8) |
| Range | 5.09 s – 5.23 s |
| **RTF** | **≈ 0.17** (~5.8× faster than real time) |

System load sampled during those runs: CPU 4.3% mean against a 2.2% idle
baseline, integrated GPU 10% against 7% idle (that delta is desktop
compositing), discrete GPU flat at 0%. The transcription work doesn't show up on
any of them — it's on the NPU.

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