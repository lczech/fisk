#!/usr/bin/env python3
"""
Conventions: each (encoding, layout) combination of a variant side by side, one panel per
CPU/compiler.

Reads the tables of one suite, or of the checked and unchecked suite of a pair (such as
kmer_extract_ascii_checked.csv and kmer_extract_ascii_unchecked.csv), of one or more CPU/compiler
directories, the tables of a directory together. Averages over the other case fields (k, masks),
and draws a grid of grouped bar charts: x-axis = technique, one bar per combination of the
conventions that the suite varies, which are the encoding and the layout, or one of them. Only
techniques benchmarked with at least two combinations are shown, as the others have nothing to
compare here; directories without any such technique (e.g. results from before the encoding and
layout were recorded) are skipped. The spaced suites keep both layouts here, unlike in the other
plots, see normalize_spaced_layout().

The panel label is taken from the last path component of the directory containing each CSV, as in
plot_bars_per_cpu.py, e.g. "results/AMD EPYC 9684X, Clang 17/kmer_extract_ascii_checked.csv" ->
"AMD EPYC 9684X, Clang 17".

Usage examples:
  python plot/plot_conventions_per_cpu.py \
      --glob "results/*/kmer_extract_ascii_*checked.csv" \
      --out results/Summaries/kmer_extract_ascii_conventions_per_cpu.png

  python plot/plot_conventions_per_cpu.py \
      --file "results/AMD Ryzen 7 Pro 4750U, GCC 14/sequence_pack_checked.csv" --unit ns

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


# The conventions, in the order of the bars. Rows without a field have the first of its values,
# as for result files from before the field was recorded.
FIELDS = {
    "encoding": ["acgt", "actg"],
    "layout": ["msb", "lsb"],
}

# Hue by encoding, shade by layout, for the full combination, also where a suite varies only one.
COMBO_COLORS = {
    ("acgt", "msb"): "#4C72B0",
    ("acgt", "lsb"): "#A1B8DC",
    ("actg", "msb"): "#DD8452",
    ("actg", "lsb"): "#F0BFA0",
}


def combo_axes(fields):
    """The bar keys, labels and colors of the combinations of the given fields, in plot order."""
    axes, labels, colors = [], {}, {}
    for encoding in FIELDS["encoding"]:
        for layout in FIELDS["layout"]:
            values = {"encoding": encoding, "layout": layout}
            key = "_".join(values[f] for f in fields)
            if key in labels:
                continue
            axes.append(key)
            labels[key] = " / ".join(values[f].upper() for f in fields)
            colors[key] = COMBO_COLORS[(encoding, layout)]
    return axes, labels, colors


def load_tables(csv_paths, mode):
    """The rows of the tables of one directory, of the variants of `mode`, with their case split
    into columns, and the convention fields that they vary."""
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = filter_mode(parse_case_fields(df), mode)
    fields = [f for f in FIELDS if f in df.columns and df[f].nunique() > 1]
    for field, values in FIELDS.items():
        if field not in df.columns:
            df[field] = values[0]
        df[field] = df[field].fillna(values[0])
    return df, fields


def load_technique_combo_pivot(csv_paths, unit, mode, fields):
    """
    Load the CSVs of one directory and return (suite, pivot), with pivot index = technique,
    columns = the combinations of `fields` present, values in the display unit. Only techniques
    with at least two combinations are kept. The mean over the other case fields is taken in ns/op
    space, and only then converted, see convert_for_display().
    """
    df, _ = load_tables(csv_paths, mode)
    suite = single_suite(df)
    if suite is None:
        raise ValueError(f"Expected the tables of one suite in {csv_paths}")

    df["combo"] = df[fields].agg("_".join, axis=1) if fields else ""
    pivot = df.pivot_table(index="benchmark", columns="combo", values="ns_per_op", aggfunc="mean")
    pivot = pivot[pivot.notna().sum(axis=1) >= 2]
    pivot = convert_for_display(pivot, unit)

    axes, _, _ = combo_axes(fields)
    ordered = [b for b in BENCHMARK_ORDER if b in pivot.index]
    ordered += [b for b in pivot.index if b not in ordered]
    pivot = pivot.reindex(index=ordered, columns=[a for a in axes if a in pivot.columns])
    return suite, pivot


def main():
    ap = argparse.ArgumentParser(
        description="Variants per (encoding, layout) combination, one panel per CPU."
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
    add_mode_arg(ap)
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

    # The conventions that any of the directories vary.
    fields = []
    for paths in groups.values():
        fields += [f for f in load_tables(paths, args.mode)[1] if f not in fields]
    fields = [f for f in FIELDS if f in fields]
    # Not an error: a suite with one encoding and layout has nothing to compare here.
    if not fields:
        warn(f"Nothing to plot: the tables {files} have a single encoding and layout")
        return

    def load(paths, unit):
        return load_technique_combo_pivot(paths, unit, args.mode, fields)

    usable = []
    for paths in groups.values():
        if load(paths, args.unit)[1].empty:
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

    _, labels, colors = combo_axes(fields)
    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        usable, load, labels, colors,
        "per " + " and ".join(fields),
        out, args.unit, args.scale, args.y_min, args.y_max, args.title,
        xtick_label=lambda t: label_for_mode(t, args.mode),
    )


if __name__ == "__main__":
    main()
