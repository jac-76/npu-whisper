#!/usr/bin/env python3
"""Benchmark whisper-npu transcription on the NPU.

Times real `whisper-npu` invocations against an audio file and reports
mean/stddev + real-time factor (RTF). Takes the audio length from the file
itself (via ffprobe) so the RTF is honest and reproducible.

Usage:
  bench [--runs N] [--warmup SKIP] [--power] AUDIO_FILE
  e.g. bench --runs 10 --warmup 2 --power sample.ogg

The first run includes server-warm latency if the FLM server isn't already up;
subsequent runs are warm. Both are reported separately.

--power additionally reads the RAPL energy counters
(/sys/class/powercap/intel-rapl:0) around each run and reports average CPU
package + core power and energy per transcription. Needs those counters to be
readable (they are, by default, on most AMD/Intel Linux systems).
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHISPER_NPU = HERE / "whisper-npu"

RAPL_PKG = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_CORE = Path("/sys/class/powercap/intel-rapl:0:0/energy_uj")


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


def rapl_ok() -> bool:
    try:
        int(RAPL_PKG.read_text())
        return True
    except OSError:
        return False


def _read(p: Path) -> int:
    return int(p.read_text())


def run_once(path: Path, power: bool):
    """Return (seconds, pkg_joules, core_joules). Joules are None if power=False."""
    if power:
        e0, k0 = _read(RAPL_PKG), _read(RAPL_CORE)
    t0 = time.perf_counter()
    subprocess.run(
        [str(WHISPER_NPU), "--json", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    dt = time.perf_counter() - t0
    if not power:
        return dt, None, None
    return dt, (_read(RAPL_PKG) - e0) / 1e6, (_read(RAPL_CORE) - k0) / 1e6


def idle_power(seconds: float):
    """Average pkg/core watts over `seconds` of doing nothing."""
    e0, k0, t0 = _read(RAPL_PKG), _read(RAPL_CORE), time.perf_counter()
    time.sleep(seconds)
    dt = time.perf_counter() - t0
    return (_read(RAPL_PKG) - e0) / 1e6 / dt, (_read(RAPL_CORE) - k0) / 1e6 / dt


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
    p.add_argument("--power", action="store_true",
                   help="also measure CPU package/core power via RAPL")
    p.add_argument("audio", type=Path)
    args = p.parse_args()

    if args.runs < 2:
        p.error("--runs must be >= 2 (need multiple samples for stdev)")
    if not args.audio.is_file():
        p.error(f"audio file not found: {args.audio}")
    if args.warmup >= args.runs:
        p.error("--warmup must be < --runs")

    power = args.power
    if power and not rapl_ok():
        print(f"warning: {RAPL_PKG} not readable; --power disabled", file=sys.stderr)
        power = False

    duration = audio_duration(args.audio)
    print(f"audio: {args.audio} ({duration:.1f}s)")
    print(f"runs: {args.runs} (skipping first {args.warmup} as warm-up)")

    idle_pkg = idle_core = None
    if power:
        idle_pkg, idle_core = idle_power(10.0)
        print(f"idle: pkg={idle_pkg:.2f} W  core={idle_core:.2f} W")
    print()

    all_times, pkg_j, core_j = [], [], []
    for i in range(args.runs):
        t, pj, cj = run_once(args.audio, power)
        all_times.append(t)
        line = f"  run {i + 1}: {t:.3f}s"
        if power:
            pkg_j.append(pj)
            core_j.append(cj)
            line += f"   pkg {pj:.1f} J ({pj / t:.1f} W)   core {cj:.1f} J ({cj / t:.1f} W)"
        print(line)

    print()
    warm = all_times[args.warmup:]
    summarize("warm runs", warm, duration)
    if len(all_times) > 1:
        summarize("all runs", all_times, duration)

    if power:
        wj_pkg = pkg_j[args.warmup:]
        wj_core = core_j[args.warmup:]
        ws = warm
        mean_s = statistics.fmean(ws)
        mean_pkg = statistics.fmean(wj_pkg)
        mean_core = statistics.fmean(wj_core)
        print()
        print(f"power (warm): pkg {mean_pkg / mean_s:.1f} W  core {mean_core / mean_s:.1f} W "
              f"(idle pkg {idle_pkg:.1f} W, core {idle_core:.1f} W)")
        print(f"energy/transcription: {mean_pkg:.0f} J total, "
              f"{mean_pkg - idle_pkg * mean_s:.0f} J over idle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
