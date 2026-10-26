#!/usr/bin/env python3
"""
K-mer extraction: each (encoding, layout) combination side by side, one panel per CPU/compiler.

Reads the kmer_extract_ascii_checked.csv and kmer_extract_ascii_unchecked.csv files of one or more
CPU/compiler directories, the two tables of a directory together, averages over k, and draws a
grid of grouped bar charts: x-axis = technique, one bar per (encoding, layout) combination.
Only techniques benchmarked with at least two combinations are shown, as the others have nothing
to compare here; files without any such technique (e.g. results from before the encoding and
layout were recorded) are skipped.

The panel label is taken from the last path component of the directory containing each CSV, as in
plot_bars_per_cpu.py, e.g. "results/AMD EPYC 9684X, Clang 17/kmer_extract_ascii_checked.csv" ->
"AMD EPYC 9684X, Clang 17".

Usage examples:
  python plot/plot_kmer_extract_ascii_axis_per_cpu.py \
      --glob "results/*/kmer_extract_ascii_*checked.csv" \
      --out results/Summaries/kmer_extract_ascii_axis_per_cpu.png

  python plot/plot_kmer_extract_ascii_axis_per_cpu.py \
      --file "results/AMD Ryzen 7 Pro 4750U, GCC 14/kmer_extract_ascii_checked.csv" --unit ns

If --out is omitted, shows interactively.
"""

import argparse
import glob
import os
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


def load_technique_combo_pivot(csv_paths, unit):
    """
    Load the CSVs of one directory and return (suite, pivot), with pivot index = technique,
    columns = the (encoding, layout) combinations present, values in the display unit. Only
    techniques with at least two combinations are kept. The mean over k is taken in ns/op space, and only then
    converted, see convert_for_display().
    """
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    suite = single_suite(df)
    if suite is None:
        raise ValueError(f"Expected the tables of one suite in {csv_paths}")

    # Rows without an encoding or layout field are ACGT and MSB, as in plot_kmer_extract_ascii.py.
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
    return suite, pivot


def main():
    ap = argparse.ArgumentParser(
        description="K-mer extraction per (encoding, layout) combination, one panel per CPU."
    )
    ap.add_argument(
        "--file", action="append", default=[],
        help="Input CSV (repeatable); the CSVs of one CPU/compiler directory go into one panel.",
    )
    ap.add_argument(
        "--glob", action="append", default=[],
        help='Glob pattern for input CSVs (repeatable), e.g. "results/*/kmer_extract_ascii_*.csv".',
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

    # The tables of one directory go into one panel.
    groups = {}
    for path in files:
        groups.setdefault(os.path.dirname(path), []).append(path)

    usable = []
    for paths in groups.values():
        if load_technique_combo_pivot(paths, args.unit)[1].empty:
            warn(f"Skipping {paths}: no technique with more than one encoding/layout combination")
        else:
            usable.append(paths)
    # Not an error: expected for result files from before encoding and layout were recorded.
    if not usable:
        warn(
            "Nothing to plot: no input file has a technique with several encoding/layout "
            "combinations"
        )
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
