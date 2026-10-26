#!/usr/bin/env python3
"""
SIMD extraction: the vectors that a technique hands out vs. its single k-mers, taken out of the
vectors one at a time, side by side, one panel per CPU/compiler.

Reads one or more CSV files of one suite (one per CPU/compiler directory), such as
spaced_kmer_extract_ascii_checked_single.csv or kmer_extract_packed_checked.csv, keeps the SIMD
rows that come in both forms, averages over cases (masks, mask sets, k, layouts), and draws a grid
of grouped bar charts: x-axis = technique, two bars per technique = its two forms. The single
k-mers are the rows with the "_scalars" suffix; the vectors are those with the "_vectors" suffix,
or, where a suite names them without a suffix (such as "bytewise_simd_avx2" in
kmer_extract_packed_checked), the same name without "_scalars". Techniques with only one of the two
forms are dropped.

The panel label is taken from the last path component of the directory containing each CSV, as in
plot_bars_per_cpu.py, e.g.
"results/AMD EPYC 9684X, Clang 17/spaced_kmer_extract_ascii_checked_multi.csv" ->
"AMD EPYC 9684X, Clang 17".

Usage examples:
  python plot/plot_vectors_scalars_per_cpu.py \
      --glob "results/*/spaced_kmer_extract_ascii_checked_multi.csv" \
      --out \
        results/Summaries/spaced_kmer_extract_ascii_checked_multi_vectors_scalars_per_cpu.png

  python plot/plot_vectors_scalars_per_cpu.py \
      --file "results/AMD Ryzen 7 Pro 4750U, GCC 14/spaced_kmer_extract_ascii_checked_single.csv" \
      --unit ns

If --out is omitted, shows interactively.
"""

import argparse
import glob
import re
import sys

import pandas as pd

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


FORMS = ["vectors", "scalars"]

FORM_LABELS = {
    "vectors": "vectors",
    "scalars": "single k-mers (scalars)",
}

FORM_COLORS = {
    "vectors": "#4C72B0",
    "scalars": "#DD8452",
}


def _technique_forms(benchmarks):
    """
    Map each benchmark name that comes in both forms to its (technique, form), see the module
    docstring. Names of only one form are left out.
    """
    names = set(benchmarks)
    result = {}
    for name in names:
        if name.endswith("_scalars"):
            technique = name.removesuffix("_scalars")
            if f"{technique}_vectors" in names or technique in names:
                result[name] = (technique, "scalars")
        elif name.endswith("_vectors"):
            technique = name.removesuffix("_vectors")
            if f"{technique}_scalars" in names:
                result[name] = (technique, "vectors")
        elif f"{name}_scalars" in names:
            result[name] = (name, "vectors")
    return result


def load_csv(csv_path):
    """Load one CSV, with the spaced suites restricted to MSB, see normalize_spaced_layout()."""
    return normalize_spaced_layout(pd.read_csv(csv_path))


def has_form_rows(csv_path):
    """Whether a CSV has any technique in both forms; older result files do not."""
    return bool(_technique_forms(load_csv(csv_path)["benchmark"]))


def load_technique_form_pivot(csv_path, unit):
    """
    Load one CSV and return (suite, pivot), with pivot index = technique, columns = FORMS, values
    in the display unit. The mean over cases is taken in ns/op space, and only then converted, see
    convert_for_display().
    """
    df = load_csv(csv_path)
    forms = _technique_forms(df["benchmark"])
    df = df[df["benchmark"].isin(forms.keys())].copy()
    if df.empty:
        raise ValueError(f"No techniques with both vectors and scalars in {csv_path}")

    suites = df["suite"].unique()
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one suite in {csv_path}, found {list(suites)}")

    df["technique"] = df["benchmark"].map(lambda b: forms[b][0])
    df["form"] = df["benchmark"].map(lambda b: forms[b][1])

    pivot = df.pivot_table(index="technique", columns="form", values="ns_per_op", aggfunc="mean")
    pivot = convert_for_display(pivot, unit)

    # The order of the vectors in BENCHMARK_ORDER, then any others alphabetically.
    order = [b.removesuffix("_vectors") for b in BENCHMARK_ORDER if b.endswith("_vectors")]
    ordered = [t for t in order if t in pivot.index]
    ordered += sorted(t for t in pivot.index if t not in ordered)
    pivot = pivot.reindex(index=ordered, columns=[f for f in FORMS if f in pivot.columns])
    return suites[0], pivot


def main():
    ap = argparse.ArgumentParser(
        description="SIMD extraction, vectors vs. single k-mers, one panel per CPU."
    )
    ap.add_argument(
        "--file", action="append", default=[],
        help="Input CSV (repeatable), one per CPU/compiler directory.",
    )
    ap.add_argument(
        "--glob", action="append", default=[],
        help=(
            'Glob pattern for input CSVs (repeatable), e.g. '
            '"results/*/spaced_kmer_extract_ascii_checked_multi.csv".'
        ),
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
        if has_form_rows(path):
            usable.append(path)
        else:
            warn(f"Skipping {path}: no techniques with both vectors and scalars")
    # Not an error: expected for result files from before both forms were recorded.
    if not usable:
        warn("Nothing to plot: no input file has techniques with both vectors and scalars")
        return

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        usable, load_technique_form_pivot, FORM_LABELS, FORM_COLORS,
        "SIMD extraction, vectors vs. single k-mers",
        out, args.unit, args.scale, args.y_min, args.y_max, args.title,
        xtick_label=lambda t: re.sub(r"^(rolling|bytewise)_simd_", "", t),
    )


if __name__ == "__main__":
    main()
