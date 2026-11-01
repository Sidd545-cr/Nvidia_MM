#!/usr/bin/env python3
"""
run_tests.py -- build the four versions (CPU, naive GPU, tiled GPU, padded tiled GPU),
run every test case, and check each output against the correct result to the first
2 decimal places.  The tiled and padded versions run with tile size 16 by default.

Usage:  python3 tests/run_tests.py            normal cases, tile size 16
        python3 tests/run_tests.py --32       tile size 32 instead of 16
        python3 tests/run_tests.py --all      both tile sizes, 16 and 32
        python3 tests/run_tests.py --large    also 1024^3, 2048x512x4096 and 2048^3 (first run
                                              generates them, which takes a few minutes)
        (--large can be combined with the others, e.g. --all --large)
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
TILED_BIN = BUILD / "tiled_fixed_gpu"
PADDED_BIN = BUILD / "padded_gpu"
# Tile sizes passed to the tiled and padded programs.
if "--all" in sys.argv:
    TILES = ("16", "32")
elif "--32" in sys.argv:
    TILES = ("32",)
else:
    TILES = ("16",)


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
subprocess.run(["nvcc", "-O3", str(ROOT / "tiled_fixed_gpu.cu"), "-o", str(TILED_BIN)], check=True)
subprocess.run(["nvcc", "-O3", str(ROOT / "padded_gpu.cu"), "-o", str(PADDED_BIN)], check=True)


def gpu_cols(res, out):
    return (f"{res:<10} {get_time(out, 'GPU kernel'):>9} "
            f"{get_time(out, 'GPU end-to-end'):>8}   ")


def winner(names, results, times):
    """Name of the fastest version that passed (CPU time vs GPU kernel times)."""
    best = "-"
    best_ms = None
    for n, r, t in zip(names, results, times):
        try:
            ms = float(t)
        except ValueError:
            continue
        if r == "PASS" and (best_ms is None or ms < best_ms):
            best, best_ms = n, ms
    return best


cases = gen_cases.CASES + (gen_cases.LARGE_CASES if "--large" in sys.argv else [])

# Versions in table order; the tiled and padded programs take the tile size as argument.
VERSIONS = ([("Naive", GPU_BIN, [])]
            + [(f"Tiled {t}", TILED_BIN, [t]) for t in TILES]
            + [(f"Padded {t}", PADDED_BIN, [t]) for t in TILES])
NAMES = ["CPU"] + [label for label, _, _ in VERSIONS]
WIDTH = 57 + 32 * len(VERSIONS) + 10
print(f"{'Case':<19} {'M x N x L':<15} {'CPU':<10} {'CPU ms':>8}   "
      + "".join(f"{label:<10} {'kernel ms':>9} {'e2e ms':>8}   " for label in NAMES[1:])
      + "Fastest")
print("-" * WIDTH)

all_ok = True
times = []   # per case: (name, CPU ms, [(tiled kernel ms, tiled e2e ms) for each tile size])
for name, M, N, L, _ in cases:
    gen_cases.generate(name, M, N, L)
    d = gen_cases.CASES_DIR / name
    expected = read_values(d / "expected.txt")

    cpu, cpu_out = run(CPU_BIN, d, "cpu_out.txt")
    cpu_res = check(cpu, expected)
    line = f"{name:<19} {f'{M}x{N}x{L}':<15} {cpu_res:<10} {get_time(cpu_out, 'CPU'):>8}   "
    results, compute_ms, tiled_times = [cpu_res], [get_time(cpu_out, "CPU")], []

    for label, binary, extra in VERSIONS:
        got, out = run(binary, d, label.lower().replace(" ", "") + "_out.txt", extra)
        res = check(got, expected)
        line += gpu_cols(res, out)
        results.append(res)
        compute_ms.append(get_time(out, "GPU kernel"))
        if label.startswith("Tiled"):
            tiled_times.append((get_time(out, "GPU kernel"), get_time(out, "GPU end-to-end")))

    all_ok &= all(r == "PASS" for r in results)
    print(line + winner(NAMES, results, compute_ms))
    times.append((name, get_time(cpu_out, "CPU"), tiled_times))

print("-" * WIDTH)
print("All tests passed." if all_ok else "Some tests FAILED.")


def speedup(cpu_ms, gpu_ms):
    """CPU time / GPU time as text ('-' if a time is missing or 0)."""
    try:
        c, g = float(cpu_ms), float(gpu_ms)
    except ValueError:
        return "-"
    return f"{c / g:.2f}x" if c > 0 and g > 0 else "-"


# Speedup of the tiled GPU version over the CPU.
# Above 1x the GPU is faster, below 1x the CPU is faster.
# "kernel" compares the computation only; "e2e" includes GPU allocation and copies.
SP_WIDTH = 20 + 32 * len(TILES)
print("\nSpeedup of tiled GPU over CPU  (CPU time / GPU time; below 1x the CPU was faster)")
print(f"{'Case':<19} " + "".join(f"{f'Tiled {t} kernel':>17} {f'Tiled {t} e2e':>14}" for t in TILES))
print("-" * SP_WIDTH)
for name, cpu_ms, tiled_times in times:
    print(f"{name:<19} " + "".join(f"{speedup(cpu_ms, k):>17} {speedup(cpu_ms, e):>14}"
                                   for k, e in tiled_times))
print("-" * SP_WIDTH)

sys.exit(0 if all_ok else 1)
