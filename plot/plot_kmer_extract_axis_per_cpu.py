#!/usr/bin/env python3
"""
K-mer extraction: each (encoding, layout) combination side by side, one panel per CPU/compiler.

Reads one or more kmer_extract.csv files (one per CPU/compiler directory), averages over k, and
draws a grid of grouped bar charts: x-axis = technique, one bar per (encoding, layout) combination.
Only techniques benchmarked with at least two combinations are shown, as the others have nothing
to compare here; files without any such technique (e.g. results from before the encoding and
layout were recorded) are skipped.

The panel label is taken from the last path component of the directory containing each CSV, as in
plot_bars_per_cpu.py, e.g. "results/AMD EPYC 9684X, Clang 17/kmer_extract.csv" ->
"AMD EPYC 9684X, Clang 17".

Usage examples:
  python plot/plot_kmer_extract_axis_per_cpu.py \
      --glob "results/*/kmer_extract.csv" \
      --out results/Summaries/kmer_extract_axis_per_cpu.png

  python plot/plot_kmer_extract_axis_per_cpu.py \
      --file "results/AMD Ryzen 7 Pro 4750U, GCC 14/kmer_extract.csv" --unit ns

If --out is omitted, shows interactively.
"""

import argparse
import glob
import sys

import pandas as pd

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


AXES = ["acgt_msb", "acgt_lsb", "actg_msb", "actg_lsb"]

AXIS_LABELS = {
    "acgt_msb": "ACGT / MSB",
    "acgt_lsb": "ACGT / LSB",
    "actg_msb": "ACTG / MSB",
    "actg_lsb": "ACTG / LSB",
}

# Hue by encoding, shade by layout.
AXIS_COLORS = {
    "acgt_msb": "#4C72B0",
    "acgt_lsb": "#A1B8DC",
    "actg_msb": "#DD8452",
    "actg_lsb": "#F0BFA0",
}


def load_technique_combo_pivot(csv_path, unit):
    """
    Load one CSV and return (suite, pivot), with pivot index = technique, columns = the
    (encoding, layout) combinations present, values in the display unit. Only techniques with at
    least two combinations are kept. The mean over k is taken in ns/op space, and only then
    converted, see convert_for_display().
    """
    df = pd.read_csv(csv_path)
    suites = df["suite"].unique()
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one suite in {csv_path}, found {list(suites)}")

    # Rows without an encoding or layout field are ACGT and MSB, as in plot_kmer_extract.py.
    df = parse_case_fields(df)
    for field, default in [("encoding", "acgt"), ("layout", "msb")]:
        if field not in df.columns:
            df[field] = default
        df[field] = df[field].fillna(default)
    df["combo"] = df["encoding"] + "_" + df["layout"]

    pivot = df.pivot_table(index="benchmark", columns="combo", values="ns_per_op", aggfunc="mean")
    pivot = pivot[pivot.notna().sum(axis=1) >= 2]
    pivot = convert_for_display(pivot, unit)

    ordered = [b for b in BENCHMARK_ORDER if b in pivot.index]
    ordered += [b for b in pivot.index if b not in ordered]
    pivot = pivot.reindex(index=ordered, columns=[a for a in AXES if a in pivot.columns])
    return suites[0], pivot


def main():
    ap = argparse.ArgumentParser(
        description="K-mer extraction per (encoding, layout) combination, one panel per CPU."
    )
    ap.add_argument(
        "--file", action="append", default=[],
        help="Input CSV (repeatable), one per CPU/compiler directory.",
    )
    ap.add_argument(
        "--glob", action="append", default=[],
        help='Glob pattern for input CSVs (repeatable), e.g. "results/*/kmer_extract.csv".',
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
        if load_technique_combo_pivot(path, args.unit)[1].empty:
            print(f"Skipping {path}: no technique with more than one encoding/layout combination")
        else:
            usable.append(path)
    # Not an error: expected for result files from before encoding and layout were recorded.
    if not usable:
        print("Nothing to plot.")
        return

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        usable, load_technique_combo_pivot, AXIS_LABELS, AXIS_COLORS,
        "k-mer extraction per encoding and layout",
        out, args.unit, args.scale, args.y_min, args.y_max, args.title,
        xtick_label=lambda t: BENCHMARK_RENAMES.get(t, t),
    )


if __name__ == "__main__":
    main()
