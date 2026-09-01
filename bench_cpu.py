#!/usr/bin/env python3
"""CPU baseline for the NPU benchmark: whisper.cpp on the same clip + model.

Companion to `bench.py`. Times `whisper-cli` (whisper.cpp) transcribing an audio
file on the CPU and reads the RAPL energy counters around every run, so the two
scripts produce directly comparable "energy per transcription" numbers.

Usage:
  bench_cpu.py --whisper-cli PATH --model ggml-large-v3-turbo.bin \\
               [--threads 16] [--runs 5] [--warmup 1] audio.wav

Notes:
  * whisper.cpp wants a 16 kHz mono WAV:
      ffmpeg -i clip.ogg -ar 16000 -ac 1 -c:a pcm_s16le clip.wav
  * The Arch `whisper-cpp` package's ggml backend is currently broken
    (GGML_ASSERT(device) at startup); build from source instead:
      git clone --depth 1 https://github.com/ggerganov/whisper.cpp
      cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    then point --whisper-cli at build/bin/whisper-cli.
  * Model: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

RAPL_PKG = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_CORE = Path("/sys/class/powercap/intel-rapl:0:0/energy_uj")


def _read(p: Path) -> int:
    return int(p.read_text())


def rapl_ok() -> bool:
    try:
        _read(RAPL_PKG)
        return True
    except OSError:
        return False


def audio_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def idle_power(seconds: float):
    e0, k0, t0 = _read(RAPL_PKG), _read(RAPL_CORE), time.perf_counter()
    time.sleep(seconds)
    dt = time.perf_counter() - t0
    return (_read(RAPL_PKG) - e0) / 1e6 / dt, (_read(RAPL_CORE) - k0) / 1e6 / dt


def run_once(cli: str, model: str, wav: Path, threads: int):
    e0, k0 = _read(RAPL_PKG), _read(RAPL_CORE)
    t0 = time.perf_counter()
    subprocess.run(
        [cli, "-m", model, "-f", str(wav), "-t", str(threads), "-np", "-nt"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    dt = time.perf_counter() - t0
    return dt, (_read(RAPL_PKG) - e0) / 1e6, (_read(RAPL_CORE) - k0) / 1e6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whisper-cli", required=True, help="path to whisper-cli binary")
    p.add_argument("--model", required=True, help="path to ggml-*.bin model")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("audio", type=Path, help="16 kHz mono WAV")
    args = p.parse_args()

    if args.runs < 2 or args.warmup >= args.runs:
        p.error("need --runs >= 2 and --warmup < --runs")
    if not args.audio.is_file():
        p.error(f"audio not found: {args.audio}")
    if not rapl_ok():
        p.error(f"{RAPL_PKG} not readable — can't measure power")

    dur = audio_duration(args.audio)
    print(f"audio: {args.audio} ({dur:.1f}s)   threads: {args.threads}   "
          f"runs: {args.runs} (warmup {args.warmup})")
    ip, ic = idle_power(10.0)
    print(f"idle: pkg={ip:.2f} W  core={ic:.2f} W\n")

    secs, pkgj, corej = [], [], []
    for i in range(args.runs):
        dt, pj, cj = run_once(args.whisper_cli, args.model, args.audio, args.threads)
        secs.append(dt); pkgj.append(pj); corej.append(cj)
        print(f"  run {i + 1}: {dt:6.2f} s   pkg {pj:6.1f} J ({pj / dt:5.1f} W)   "
              f"core {cj:6.1f} J ({cj / dt:5.1f} W)")

    ws, wp, wc = secs[args.warmup:], pkgj[args.warmup:], corej[args.warmup:]
    ms, mp, mc = statistics.fmean(ws), statistics.fmean(wp), statistics.fmean(wc)
    print(f"\nwarm n={len(ws)}  wall {ms:.2f}s  stdev {statistics.pstdev(ws):.2f}s"
          + (f"  RTF {ms / dur:.3f}" if dur else ""))
    print(f"power (warm): pkg {mp / ms:.1f} W  core {mc / ms:.1f} W "
          f"(idle pkg {ip:.1f} W, core {ic:.1f} W)")
    print(f"energy/transcription: {mp:.0f} J total, {mp - ip * ms:.0f} J over idle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
