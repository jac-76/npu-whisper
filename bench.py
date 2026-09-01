#!/usr/bin/env python3
"""Benchmark whisper-npu transcription on the NPU.

Times real `whisper-npu` invocations against an audio file and reports
mean/stddev + real-time factor (RTF). Takes the audio length from the file
itself (via ffprobe) so the RTF is honest and reproducible.

Usage:
  bench [--runs N] [--warmup SKIP] AUDIO_FILE
  e.g. bench --runs 5 sample.ogg

The first run includes server-warm latency if the FLM server isn't already up;
subsequent runs are warm. Both are reported separately.
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHISPER_NPU = HERE / "whisper-npu"


def audio_duration(path: Path) -> float:
    """Seconds of audio via ffprobe; falls back to 0.0 if ffprobe is missing."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        print("warning: ffprobe unavailable; RTF will be omitted", file=sys.stderr)
        return 0.0


def run_once(path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [str(WHISPER_NPU), "--json", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return time.perf_counter() - t0


def summarize(label: str, times: list[float], duration: float) -> None:
    if not times:
        return
    mean = statistics.fmean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"{label}: n={len(times)}  mean={mean:.3f}s  stdev={sd:.3f}s  "
          f"min={min(times):.3f}s  max={max(times):.3f}s")
    if duration > 0:
        print(f"  {label.lower()} RTF = mean/duration = {mean:.3f}s / {duration:.1f}s "
              f"= {mean / duration:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1,
                   help="skip the first N runs (server warm-up)")
    p.add_argument("audio", type=Path)
    args = p.parse_args()

    if args.runs < 2:
        p.error("--runs must be >= 2 (need multiple samples for stdev)")
    if not args.audio.is_file():
        p.error(f"audio file not found: {args.audio}")
    if args.warmup >= args.runs:
        p.error("--warmup must be < --runs")

    duration = audio_duration(args.audio)
    print(f"audio: {args.audio} ({duration:.1f}s)")
    print(f"runs: {args.runs} (skipping first {args.warmup} as warm-up)")
    print()

    all_times = []
    for i in range(args.runs):
        t = run_once(args.audio)
        all_times.append(t)
        print(f"  run {i + 1}: {t:.3f}s")

    print()
    warm = all_times[args.warmup:]
    summarize("warm runs", warm, duration)
    if len(all_times) > 1:
        summarize("all runs", all_times, duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())