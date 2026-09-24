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
import matplotlib.pyplot as plt

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


def plot_grid(csv_paths, outpath, unit, scale, y_min, y_max, title):
    entries = [(p, platform_from_csv_path(p)) for p in csv_paths]
    entries.sort(key=lambda e: platform_compiler_sort_key(e[1]))

    loaded = []
    for csv_path, label in entries:
        suite, pivot = load_technique_axis_pivot(csv_path, unit)
        loaded.append((label, suite, pivot))

    suites = {suite for _, suite, _ in loaded}
    if len(suites) != 1:
        raise ValueError(f"All input files must be from the same suite, found {sorted(suites)}")
    suite = suites.pop()

    # One shared y-range across all panels, so they are directly comparable.
    all_values = np.concatenate([pivot.values.ravel() for _, _, pivot in loaded])
    axis_y_min, axis_y_max = compute_axis_limits(all_values, scale)
    # Extra top headroom for the rotated value labels above each bar, as in plot_kmer_spaced.py.
    axis_y_max *= 1.3 if scale == "log" else 1.1
    if y_min is not None:
        axis_y_min = y_min
    if y_max is not None:
        axis_y_max = y_max

    n = len(loaded)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(8 * ncols, 12), 6 * nrows), squeeze=False, layout="constrained"
    )

    for idx, (label, _, pivot) in enumerate(loaded):
        ax = axes[idx // ncols][idx % ncols]
        apply_yscale(ax, scale)

        techniques = list(pivot.index)
        x = np.arange(len(techniques))
        axis_names = list(pivot.columns)
        bar_width = 0.8 / max(1, len(axis_names))
        offsets = [(j - (len(axis_names) - 1) / 2.0) * bar_width for j in range(len(axis_names))]

        for j, axis_name in enumerate(axis_names):
            vals = pivot[axis_name].values
            bars = ax.bar(
                x + offsets[j], np.nan_to_num(vals, nan=0.0), width=bar_width,
                label=AXIS_LABELS[axis_name], color=AXIS_COLORS[axis_name],
            )
            # Multiplicative headroom on "log", additive on "linear", as in plot_kmer_spaced.py.
            for bar, val in zip(bars, vals):
                if pd.notna(val):
                    label_y = (
                        val * 1.02 if scale == "log"
                        else val + (axis_y_max - axis_y_min) * 0.01
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, label_y, f"{val:.2f}",
                        ha="center", va="bottom", rotation=90, fontsize=9,
                    )

        ax.set_title(label)
        ax.set_ylabel(ylabel_for_unit(unit, suite))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.removeprefix("simd_") for t in techniques], rotation=45, ha="right"
        )
        ax.set_ylim(axis_y_min, axis_y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide unused grid cells, if any.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # One shared legend below all panels, instead of one per panel, where it would cover bar labels.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title or f"{suite}: SIMD extraction by mask vs. by position")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))

    if outpath:
        fig.savefig(outpath, dpi=300)
        print(f"Wrote {outpath}")
    else:
        plt.show()
    plt.close(fig)


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

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_grid(files, out, args.unit, args.scale, args.y_min, args.y_max, args.title)


if __name__ == "__main__":
    main()
