#!/usr/bin/env python3
"""
run_tests.py -- build the CPU and GPU programs, run every test case, and check
each output against the correct result to the first 2 decimal places.

Usage:  python3 tests/run_tests.py
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


def read_values(path):
    with open(path) as f:
        return [float(t) for t in f.read().split()]


def run(binary, case_dir, out_name):
    """Runs one program; returns (output values or None, stdout)."""
    out = case_dir / out_name
    out.unlink(missing_ok=True)
    r = subprocess.run([str(binary), str(case_dir / "input.txt"), str(out)],
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

print(f"{'Case':<13} {'M x N x L':<15} {'CPU':<17} {'CPU ms':>9}   "
      f"{'GPU':<17} {'kernel ms':>10} {'e2e ms':>9}")
print("-" * 98)

all_ok = True
for name, M, N, L, _ in gen_cases.CASES:
    gen_cases.generate(name, M, N, L)
    d = gen_cases.CASES_DIR / name
    expected = read_values(d / "expected.txt")

    cpu, cpu_out = run(CPU_BIN, d, "cpu_out.txt")
    gpu, gpu_out = run(GPU_BIN, d, "gpu_out.txt")
    cpu_res, gpu_res = check(cpu, expected), check(gpu, expected)
    all_ok &= cpu_res == gpu_res == "PASS"

    print(f"{name:<13} {f'{M}x{N}x{L}':<15} {cpu_res:<17} {get_time(cpu_out, 'CPU'):>9}   "
          f"{gpu_res:<17} {get_time(gpu_out, 'GPU kernel'):>10} "
          f"{get_time(gpu_out, 'GPU end-to-end'):>9}")

print("-" * 98)
print("All tests passed." if all_ok else "Some tests FAILED.")
sys.exit(0 if all_ok else 1)
