#!/usr/bin/env python3
"""
run_tests.py -- build the CPU, naive GPU and tiled GPU programs, run every test
case, and check each output against the correct result to the first 2 decimal places.

Usage:  python3 tests/run_tests.py            normal cases, tiled_gpu.cu (run-time tile size)
        python3 tests/run_tests.py --fixed    use tiled_fixed_gpu.cu (compile-time tile kernels)
        python3 tests/run_tests.py --large    also 1024^3, 2048x512x4096 and 2048^3 (first run
                                              generates them, which takes a few minutes)
        (options can be combined, e.g. --fixed --large)
"""
import re
import subprocess
import sys
from pathlib import Path

import gen_cases

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
CPU_BIN = BUILD / "matrix_cpu"
GPU_BIN = BUILD / "naive_gpu"
TILES = (16, 32)   # tile sizes passed to the tiled program (tiled_gpu: 8, 16, 32; --fixed: 16, 32)

# Which tiled version to test:
#   default  ignore/tiled_gpu.cu  one kernel, tile size chosen at run time
#   --fixed  tiled_fixed_gpu.cu   one compile-time kernel per tile size
if "--fixed" in sys.argv:
    TILED_SRC, TILED_LABEL = "tiled_fixed_gpu.cu", "Fixed"
else:
    TILED_SRC, TILED_LABEL = "ignore/tiled_gpu.cu", "Tiled"
TILED_BIN = BUILD / Path(TILED_SRC).stem


def read_values(path):
    with open(path) as f:
        return [float(t) for t in f.read().split()]


def run(binary, case_dir, out_name, extra=()):
    """Runs one program; returns (output values or None, stdout)."""
    out = case_dir / out_name
    out.unlink(missing_ok=True)
    r = subprocess.run([str(binary), str(case_dir / "input.txt"), str(out), *extra],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        return None, r.stdout
    return read_values(out), r.stdout


def get_time(stdout, label):
    m = re.search(rf"{label} time \(ms\):\s*([0-9.]+)", stdout)
    return m.group(1) if m else "-"


def check(got, expected):
    if got is None:
        return "CRASH"
    if len(got) != len(expected):
        return "FAIL (wrong size)"
    
    wrong = sum(not abs(g - e) < 0.005 for g, e in zip(got, expected))
    return "PASS" if wrong == 0 else f"FAIL ({wrong} wrong)"


BUILD.mkdir(exist_ok=True)
subprocess.run(["gcc", "-O3", str(ROOT / "matrix_cpu.c"), "-o", str(CPU_BIN)], check=True)
subprocess.run(["nvcc", "-O3", str(ROOT / "naive_gpu.cu"), "-o", str(GPU_BIN)], check=True)
subprocess.run(["nvcc", "-O3", str(ROOT / TILED_SRC), "-o", str(TILED_BIN)], check=True)


def gpu_cols(res, out):
    return (f"{res:<10} {get_time(out, 'GPU kernel'):>9} "
            f"{get_time(out, 'GPU end-to-end'):>8}   ")


cases = gen_cases.CASES + (gen_cases.LARGE_CASES if "--large" in sys.argv else [])

print(f"Tiled version: {TILED_SRC}\n")
WIDTH = 57 + 32 * (1 + len(TILES))
print(f"{'Case':<19} {'M x N x L':<15} {'CPU':<10} {'CPU ms':>8}   "
      f"{'Naive':<10} {'kernel ms':>9} {'e2e ms':>8}   "
      + "".join(f"{f'{TILED_LABEL} {t}':<10} {'kernel ms':>9} {'e2e ms':>8}   " for t in TILES))
print("-" * WIDTH)

all_ok = True
times = []   # per case: (name, CPU ms, [(kernel ms, e2e ms) for each tile size])
for name, M, N, L, _ in cases:
    gen_cases.generate(name, M, N, L)
    d = gen_cases.CASES_DIR / name
    expected = read_values(d / "expected.txt")

    cpu, cpu_out = run(CPU_BIN, d, "cpu_out.txt")
    gpu, gpu_out = run(GPU_BIN, d, "gpu_out.txt")
    cpu_res, gpu_res = check(cpu, expected), check(gpu, expected)
    line = (f"{name:<19} {f'{M}x{N}x{L}':<15} {cpu_res:<10} {get_time(cpu_out, 'CPU'):>8}   "
            + gpu_cols(gpu_res, gpu_out))
    all_ok &= cpu_res == gpu_res == "PASS"
    gpu_times = []

    for t in TILES:
        tiled, tiled_out = run(TILED_BIN, d, f"tiled{t}_out.txt", [str(t)])
        tiled_res = check(tiled, expected)
        all_ok &= tiled_res == "PASS"
        line += gpu_cols(tiled_res, tiled_out)
        gpu_times.append((get_time(tiled_out, "GPU kernel"), get_time(tiled_out, "GPU end-to-end")))
    print(line.rstrip())
    times.append((name, get_time(cpu_out, "CPU"), gpu_times))

print("-" * WIDTH)
print("All tests passed." if all_ok else "Some tests FAILED.")


def speedup(cpu_ms, gpu_ms):
    """CPU time / GPU time as text ('-' if a time is missing or 0)."""
    try:
        c, g = float(cpu_ms), float(gpu_ms)
    except ValueError:
        return "-"
    return f"{c / g:.2f}x" if c > 0 and g > 0 else "-"


# Speedup of the fixed-tile GPU version over the CPU (only with --fixed).
# Above 1x the GPU is faster, below 1x the CPU is faster.
# "kernel" compares the computation only; "e2e" includes GPU allocation and copies.
if "--fixed" in sys.argv:
    SP_WIDTH = 20 + 32 * len(TILES)
    print("\nSpeedup of fixed-tile GPU over CPU  (CPU time / GPU time; below 1x the CPU was faster)")
    print(f"{'Case':<19} " + "".join(f"{f'Fixed {t} kernel':>17} {f'Fixed {t} e2e':>14}" for t in TILES))
    print("-" * SP_WIDTH)
    for name, cpu_ms, gpu_times in times:
        print(f"{name:<19} " + "".join(f"{speedup(cpu_ms, k):>17} {speedup(cpu_ms, e):>14}"
                                       for k, e in gpu_times))
    print("-" * SP_WIDTH)

sys.exit(0 if all_ok else 1)
