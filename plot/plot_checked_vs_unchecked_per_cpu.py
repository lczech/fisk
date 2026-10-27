#!/usr/bin/env python3
"""
Checked vs. unchecked: the cost of the validity, per technique, side by side, one panel per
CPU/compiler.

Reads a pair of tables from each CPU/compiler directory: the unchecked suite, and its checked
counterpart with the same techniques, such as kmer_extract_packed_unchecked.csv and
kmer_extract_packed_checked.csv, or sequence_pack_unchecked.csv and sequence_pack_checked.csv. It
keeps the techniques of TECHNIQUES that both tables have, averages each over the cases that both
tables have (masks, k, layouts), and draws a grid of grouped bar charts: x-axis = technique, two
bars per technique = unchecked and checked. The spaced suites are restricted to MSB, see
normalize_spaced_layout(). Directories without both tables are left out.

The `_vectors` techniques of the checked suites sum all lanes, valid or not, so for now their
checked bars do not include the cost of using the valid lanes.

The panel label is the last path component of each directory, as in plot_bars_per_cpu.py, e.g.
"results/AMD EPYC 9684X, Clang 17".

Usage example:
  python plot/plot_checked_vs_unchecked_per_cpu.py \
      --dir "results/AMD EPYC 9684X, Clang 17" --dir "results/AMD Ryzen 7 Pro 4750U, GCC 14" \
      --unchecked kmer_extract_packed_unchecked.csv --checked kmer_extract_packed_checked.csv \
      --out results/Summaries/kmer_extract_packed_checked_vs_unchecked_per_cpu.png

If --out is omitted, shows interactively.
"""

import argparse
import os
import sys

import pandas as pd

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


FORMS = ["unchecked", "checked"]

FORM_LABELS = {
    "unchecked": "unchecked",
    "checked": "checked",
}

FORM_COLORS = {
    "unchecked": "#55A868",
    "checked": "#C44E52",
}

SIMD_ARCHS = ["sse2", "avx2", "avx512", "neon"]

# The techniques to compare, in plot order, across all pairs of suites; each pair has some of them.
# Spaced k-mers from packed sequences: the scalar loops, then the SIMD kernels, in both forms.
# Contiguous k-mers from packed sequences: the bytewise scalar loop, then the SIMD dispatchers,
# whose vectors have no suffix. Sequence packing: fisk's own encoders, without the external
# references.
TECHNIQUES = (
    [
        "bytewise_scalar_pext", "bytewise_scalar_butterfly",
        "bytewise_simd_pext_vectors", "bytewise_simd_pext_scalars",
    ]
    + [
        f"bytewise_simd_butterfly_{arch}_{form}"
        for arch in SIMD_ARCHS for form in ["vectors", "scalars"]
    ]
    + ["bytewise_scalar"]
    + [
        name for arch in SIMD_ARCHS
        for name in [f"bytewise_simd_{arch}", f"bytewise_simd_{arch}_scalars"]
    ]
    + ["direct_scalar"]
    + [name for arch in SIMD_ARCHS for name in [f"direct_{arch}", f"direct_double_{arch}"]]
)


def load_table(csv_path, mode):
    """
    Load one CSV, with the spaced suites restricted to MSB, see normalize_spaced_layout(), and
    return (suite, rows of the techniques of `mode`). TECHNIQUES are those of the supplement.
    """
    df = normalize_spaced_layout(pd.read_csv(csv_path))
    suites = df["suite"].unique()
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one suite in {csv_path}, found {list(suites)}")
    return suites[0], filter_mode(df, mode, supp=TECHNIQUES)


def load_pair_pivot(unchecked_path, checked_path, unit, mode):
    """
    Load a pair of CSVs and return (suite of the unchecked one, pivot), with pivot index =
    technique, columns = FORMS, values in the display unit. Only the (technique, case) rows that
    both tables have are kept; the mean over them is taken in ns/op space, and only then converted,
    see convert_for_display().
    """
    suite, unchecked = load_table(unchecked_path, mode)
    _, checked = load_table(checked_path, mode)
    df = pd.concat(
        [unchecked.assign(form="unchecked"), checked.assign(form="checked")], ignore_index=True
    )
    forms_per_row = df.groupby(["benchmark", "case"])["form"].transform("nunique")
    df = df[forms_per_row == len(FORMS)]

    pivot = df.pivot_table(index="benchmark", columns="form", values="ns_per_op", aggfunc="mean")
    pivot = convert_for_display(pivot, unit)
    ordered = [t for t in TECHNIQUES + BENCHMARK_ORDER if t in pivot.index]
    ordered = list(dict.fromkeys(ordered)) + sorted(t for t in pivot.index if t not in ordered)
    pivot = pivot.reindex(index=ordered, columns=[f for f in FORMS if f in pivot.columns])
    return suite, pivot


def main():
    ap = argparse.ArgumentParser(
        description="Checked vs. unchecked, per technique, one panel per CPU."
    )
    ap.add_argument(
        "--dir", action="append", default=[],
        help="CPU/compiler directory with both tables (repeatable).",
    )
    ap.add_argument("--unchecked", required=True, help="File name of the unchecked table.")
    ap.add_argument("--checked", required=True, help="File name of the checked table.")
    ap.add_argument(
        "--out", default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument(
        "--title", default=None,
        help="Figure title override (default: the name of both suites, without the check).",
    )
    add_mode_arg(ap)
    add_unit_scale_args(ap)
    args = ap.parse_args()
    if not args.dir:
        raise SystemExit("No input directories. Provide --dir ...")

    # Each panel is keyed by the path of its unchecked table, from which plot_axis_grid() takes
    # the label of its directory; the checked table is next to it.
    checked_of = {}
    for d in args.dir:
        unchecked_path = os.path.join(d, args.unchecked)
        checked_path = os.path.join(d, args.checked)
        missing = [p for p in (unchecked_path, checked_path) if not os.path.isfile(p)]
        if missing:
            warn(f"Skipping {d}: missing {', '.join(os.path.basename(p) for p in missing)}")
            continue
        _, pivot = load_pair_pivot(unchecked_path, checked_path, "ns", args.mode)
        if pivot.empty or len(pivot.columns) < len(FORMS):
            warn(f"Skipping {d}: no techniques and cases in both tables")
            continue
        checked_of[unchecked_path] = checked_path
    # Not an error: expected while a pair has not been run on any CPU yet.
    if not checked_of:
        warn(f"Nothing to plot: no directory has both {args.unchecked} and {args.checked}")
        return

    # The title names both suites, as the unchecked one without its check.
    name = os.path.splitext(os.path.basename(args.unchecked))[0].replace("_unchecked", "")
    title = args.title or f"{name}: checked vs. unchecked"

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        list(checked_of),
        lambda path, unit: load_pair_pivot(path, checked_of[path], unit, args.mode),
        FORM_LABELS, FORM_COLORS,
        "checked vs. unchecked",
        out, args.unit, args.scale, args.y_min, args.y_max, title,
    )


if __name__ == "__main__":
    main()
