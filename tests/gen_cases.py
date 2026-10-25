#!/usr/bin/env python3
"""
gen_cases.py -- create matrix-multiplication test cases and their correct results.

For every case it writes, under tests/cases/<name>/:
    input.txt     "M N L" header, then A (M x N) and B (N x L), row-major text
    expected.txt  the reference C = A * B (M x L), row-major text
    meta.json     dims + seed, so an unchanged case is not regenerated

Values are uniform in [-1, 1] from a fixed seed, written with 6 decimals.
The reference uses exactly what the C programs see: each value is parsed and
rounded to float32, then every dot product is computed with math.sumprod in
double precision.  The reference is therefore more accurate than either
program, so CPU and GPU errors are both measured against the same truth.

Usage:
    python3 tests/gen_cases.py              generate all default cases
    python3 tests/gen_cases.py --large      also the large cases (1024^3, 2048x512x4096, 2048^3)
    python3 tests/gen_cases.py --list       list cases without generating
    python3 tests/gen_cases.py --only tall  generate one case
"""
import argparse
import json
import math
import random
import sys
import time
import zlib
from array import array
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"
BASE_SEED = 2026

CASES = [
    ("tiny_1x1x1",     1,    1,    1, "smallest possible"),
    ("lt_block",       5,    7,    3, "smaller than one CUDA block"),
    ("square_64",     64,   64,   64, "square, divisible by 16"),
    ("square_256",   256,  256,  256, "square, benchmark shape"),
    ("tall",         512,   64,   32, "tall output (M >> L)"),
    ("wide",          32,   64,  512, "wide output (L >> M)"),
    ("div16",         32,   48,   64, "all dims multiples of 16"),
    ("tile_minus1",   31,   15,   47, "one less than a tile multiple"),
    ("tile_plus1",    33,   17,   49, "one more than a tile multiple"),
    ("M_eq_1",         1,  300,  200, "single output row"),
    ("L_eq_1",       200,  300,    1, "single output column"),
    ("N_eq_1",       150,    1,  120, "inner dim 1 (outer product)"),
    ("long_inner",    20, 4099,   30, "long dot products"),
    ("irregular",   1000,  513,  769, "irregular dims (spec case)"),
]
LARGE_CASES = [
    ("square_1024", 1024, 1024, 1024, "medium square (slow reference)"),
    ("rect_2048x512x4096", 2048, 512, 4096, "rectangular output (spec benchmark shape)"),
    ("square_2048", 2048, 2048, 2048, "large square (~2 min to generate)"),
]


def case_seed(name):
    return BASE_SEED ^ zlib.crc32(name.encode())


def random_matrix(rng, rows, cols):
    lines, values = [], array("f")
    for _ in range(rows):
        row = [f"{rng.uniform(-1.0, 1.0):.6f}" for _ in range(cols)]
        lines.append(" ".join(row))
        values.extend(float(s) for s in row)   # array('f') rounds to float32
    return lines, values.tolist()


def reference(M, N, L, A, B):
    Bcols = list(zip(*[B[k * L:(k + 1) * L] for k in range(N)]))
    return [[math.sumprod(A[i * N:(i + 1) * N], col) for col in Bcols]
            for i in range(M)]


def generate(name, M, N, L, force=False):
    d = CASES_DIR / name
    meta = {"M": M, "N": N, "L": L, "seed": case_seed(name)}
    meta_path = d / "meta.json"
    if not force and meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        return False   # already up to date

    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(meta["seed"])
    a_lines, A = random_matrix(rng, M, N)
    b_lines, B = random_matrix(rng, N, L)
    with open(d / "input.txt", "w") as f:
        f.write(f"{M} {N} {L}\n")
        f.write("\n".join(a_lines) + "\n")
        f.write("\n".join(b_lines) + "\n")

    C = reference(M, N, L, A, B)
    with open(d / "expected.txt", "w") as f:
        for row in C:
            f.write(" ".join(f"{v:.9g}" for v in row) + "\n")
    meta_path.write_text(json.dumps(meta))
    return True


def selected_cases(large=False, only=None):
    cases = CASES + (LARGE_CASES if large else [])
    if only:
        cases = [c for c in CASES + LARGE_CASES if c[0] in only]
        missing = set(only) - {c[0] for c in cases}
        if missing:
            sys.exit(f"unknown case(s): {', '.join(sorted(missing))}  (try --list)")
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--large", action="store_true", help="include the large cases")
    p.add_argument("--only", nargs="+", metavar="NAME", help="generate only these cases")
    p.add_argument("--force", action="store_true", help="regenerate even if up to date")
    p.add_argument("--list", action="store_true", help="list the cases and exit")
    args = p.parse_args()

    cases = selected_cases(args.large, args.only)
    if args.list:
        for name, M, N, L, desc in CASES + LARGE_CASES:
            print(f"  {name:<19} {M:>5} x {N:>5} x {L:>5}   {desc}")
        return

    for name, M, N, L, desc in cases:
        t0 = time.monotonic()
        made = generate(name, M, N, L, args.force)
        status = f"generated in {time.monotonic() - t0:.1f}s" if made else "up to date"
        print(f"  {name:<19} {M}x{N}x{L:<6} {status}")


if __name__ == "__main__":
    main()
