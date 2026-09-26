#!/usr/bin/env python3
"""
Spaced k-mer SIMD extraction: by_mask vs. by_position, side by side, one panel per CPU/compiler.

Reads one or more kmer_spaced_multi.csv / kmer_spaced_single.csv files (one per CPU/compiler
directory), keeps only the SIMD rows that carry an explicit "_by_mask"/"_by_position" suffix (see
for_each_spaced_kmer_simd_by_mask()/_by_position() in kmer_spaced/simd.hpp), averages over cases
(mask sets / masks), and draws a grid of grouped bar charts: x-axis = technique (the suffix
stripped), two bars per technique = the two emission-order axes.

plot_kmer_spaced.py splits the two axes into separate files per CSV; this puts them next to each
other instead, for every CPU/compiler at once.

The panel label is taken from the last path component of the directory containing each CSV, as in
plot_bars_per_cpu.py, e.g. "results/AMD EPYC 9684X, Clang 17/kmer_spaced_multi.csv" ->
"AMD EPYC 9684X, Clang 17".

Usage examples:
  python plot/plot_kmer_spaced_axis_per_cpu.py \
      --glob "results/*/kmer_spaced_multi.csv" \
      --out results/Summaries/kmer_spaced_multi_axis_per_cpu.png

  python plot/plot_kmer_spaced_axis_per_cpu.py \
      --file "results/AMD Ryzen 7 Pro 4750U, GCC 14/kmer_spaced_single.csv" --unit ns

If --out is omitted, shows interactively.
"""

import argparse
import glob
import sys

import numpy as np
import pandas as pd

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


# Fixed x-axis order: block-table family first, then butterfly-table, then pext; scalar -> sse2 ->
# avx2 -> avx512 -> neon within each family. Techniques absent from a given CSV (e.g. AVX512 on
# Ryzen, or anything x86 on Apple) are dropped from that panel.
TECHNIQUE_ORDER = [
    "simd_block_table_scalar",
    "simd_block_table_sse2",
    "simd_block_table_avx2",
    "simd_block_table_avx512",
    "simd_block_table_neon",
    "simd_butterfly_table_scalar",
    "simd_butterfly_table_sse2",
    "simd_butterfly_table_avx2",
    "simd_butterfly_table_avx512",
    "simd_butterfly_table_neon",
    "simd_pext",
]

AXES = ["by_mask", "by_position"]

AXIS_LABELS = {
    "by_mask": "by mask",
    "by_position": "by position",
}

AXIS_COLORS = {
    "by_mask": "#4C72B0",
    "by_position": "#DD8452",
}


def has_axis_rows(csv_path):
    """Whether a CSV has any _by_mask/_by_position rows at all; older result files do not."""
    benchmarks = pd.read_csv(csv_path)["benchmark"]
    return bool((benchmarks.str.endswith("_by_mask") | benchmarks.str.endswith("_by_position")).any())


def load_technique_axis_pivot(csv_path, unit):
    """
    Load one CSV and return (suite, pivot), with pivot index = technique (axis suffix stripped),
    columns = AXES, values in the display unit. Rows without either suffix (the axis-agnostic
    scalar/bit-extract baselines) are dropped. The mean over cases is taken in ns/op space, and
    only then converted, see convert_for_display().
    """
    df = pd.read_csv(csv_path)
    df = df[df["benchmark"].str.endswith("_by_mask") | df["benchmark"].str.endswith("_by_position")]
    if df.empty:
        raise ValueError(f"No _by_mask/_by_position rows in {csv_path}")

    suites = df["suite"].unique()
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one suite in {csv_path}, found {list(suites)}")

    df = df.copy()
    df["axis"] = np.where(df["benchmark"].str.endswith("_by_mask"), "by_mask", "by_position")
    df["technique"] = df["benchmark"].str.removesuffix("_by_mask").str.removesuffix("_by_position")

    pivot = df.pivot_table(index="technique", columns="axis", values="ns_per_op", aggfunc="mean")
    pivot = convert_for_display(pivot, unit)
    ordered = [t for t in TECHNIQUE_ORDER if t in pivot.index]
    pivot = pivot.reindex(index=ordered, columns=[a for a in AXES if a in pivot.columns])
    return suites[0], pivot


def main():
    ap = argparse.ArgumentParser(
        description="Spaced k-mer SIMD extraction, by mask vs. by position, one panel per CPU."
    )
    ap.add_argument(
        "--file", action="append", default=[],
        help="Input CSV (repeatable), one per CPU/compiler directory.",
    )
    ap.add_argument(
        "--glob", action="append", default=[],
        help='Glob pattern for input CSVs (repeatable), e.g. "results/*/kmer_spaced_multi.csv".',
    )
    ap.add_argument(
        "--out", default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument(
        "--title", default=None,
        help="Figure title override (default: derived from the suite name).",
    )
    add_unit_scale_args(ap)
    args = ap.parse_args()

    files = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise SystemExit("No input files. Provide --file ... or --glob ...")

    usable = []
    for path in files:
        if has_axis_rows(path):
            usable.append(path)
        else:
            print(f"Skipping {path}: no _by_mask/_by_position rows")
    # Not an error: expected for result files from before the two emission orders were recorded.
    if not usable:
        print("Nothing to plot.")
        return

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        usable, load_technique_axis_pivot, AXIS_LABELS, AXIS_COLORS,
        "SIMD extraction by mask vs. by position",
        out, args.unit, args.scale, args.y_min, args.y_max, args.title,
        xtick_label=lambda t: t.removeprefix("simd_"),
    )


if __name__ == "__main__":
    main()
