#!/usr/bin/env python3
"""
Spaced k-mer extraction over the span of the mask: one line per variant, one panel per
CPU/compiler.

Reads the span sweep suites, spaced_kmer_extract_<input>_<check>_spans.csv, with one mask per span
from 1 to 128, see spaced_mask_for_span() (benchmarks/spaced_masks.hpp). The spaced suites are
restricted to MSB, see normalize_spaced_layout(). Light vertical lines mark where the number of
pieces of a mask grows, see SpacedKmerMask (include/fisk/core/types.hpp): from one piece up to
span 32 to five pieces up to span 128. One CSV gives a single panel, several give a grid, with one
shared y-range.

The panel label is the CPU and compiler of each CSV, e.g. "AMD EPYC 9684X, Clang 17" for
results/AMD_EPYC_9684X/Clang_17/sum/, see platform_from_csv_path() in plot_common.py.

Usage example:
  python plot/plot_spaced_spans.py \
      --file results/AMD_EPYC_9684X/Clang_17/sum/spaced_kmer_extract_packed_checked_spans.csv \
      --file results/AMD_Ryzen_7_Pro_4750U/GCC_14/sum/spaced_kmer_extract_packed_checked_spans.csv \
      --out results/Summaries/spaced_kmer_extract_packed_checked_spans_per_cpu.png

If --out is omitted, shows interactively.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


# The largest span with one to four pieces: a first piece of up to 32 positions, then leading
# pieces of 28 positions and a tail of up to 29, see SpacedKmerMask.
PIECE_BOUNDARIES = [32, 57, 85, 113]

SIMD_ARCHS = ["scalar", "sse2", "avx2", "avx512", "neon"]


def line_style(name):
    """
    The color and line style of a variant. BENCHMARK_COLORS gives the SIMD butterfly variants of
    all archs one color, as the bar plots name the arch on the x-axis; here, they take the color of
    their arch instead, that of the chunked k-mer variants of BENCHMARK_COLORS. The SIMD PEXT
    variant is dashed, in the color of scalar PEXT.
    """
    for arch in SIMD_ARCHS:
        if name.endswith(f"_simd_butterfly_{arch}_vectors"):
            return BENCHMARK_COLORS[f"chunked_simd_{arch}"], "solid"
    if name.endswith("_simd_pext_vectors"):
        return BENCHMARK_COLORS.get(name, "black"), "dashed"
    return BENCHMARK_COLORS.get(name, "black"), "solid"


def load_table(csv_path, mode, unit):
    """Load one CSV, restricted to MSB and to the variants of `mode`, and return (suite, rows with
    an integer "span" and the "display_value" in the display unit)."""
    df = normalize_spaced_layout(pd.read_csv(csv_path))
    suites = df["suite"].unique()
    if len(suites) != 1:
        raise ValueError(f"Expected exactly one suite in {csv_path}, found {list(suites)}")
    df = filter_mode(df, mode)
    df = parse_case_fields(df)
    df["span"] = df["span"].astype(int)
    df["display_value"] = convert_for_display(df["ns_per_op"], unit)
    return suites[0], df


def main():
    ap = argparse.ArgumentParser(
        description="Spaced k-mer extraction over the span, one panel per CPU."
    )
    ap.add_argument(
        "--file", action="append", default=[],
        help="Span sweep CSV of one CPU/compiler (repeatable).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument(
        "--title", default=None, help="Figure title override (default: the suite name).",
    )
    add_mode_arg(ap)
    add_unit_scale_args(ap)
    args = ap.parse_args()
    if not args.file:
        raise SystemExit("No input files. Provide --file ...")

    entries = [(p, platform_from_csv_path(p)) for p in args.file]
    entries.sort(key=lambda e: platform_compiler_sort_key(e[1]))
    loaded = [(label, *load_table(path, args.mode, args.unit)) for path, label in entries]
    suites = {suite for _, suite, _ in loaded}
    if len(suites) != 1:
        raise ValueError(f"All input files must be from the same suite, found {sorted(suites)}")
    suite = suites.pop()

    # One shared y-range across all panels, so they are directly comparable.
    all_values = np.concatenate([df["display_value"].values for _, _, df in loaded])
    y_min, y_max = compute_axis_limits(all_values, args.scale)
    if args.y_min is not None:
        y_min = args.y_min
    if args.y_max is not None:
        y_max = args.y_max

    n = len(loaded)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False, layout="constrained",
    )

    # The variants in the order of BENCHMARK_ORDER, then any others in their order of appearance.
    present = list(dict.fromkeys(b for _, _, df in loaded for b in df["benchmark"]))
    names = [b for b in BENCHMARK_ORDER if b in present] + [
        b for b in present if b not in BENCHMARK_ORDER
    ]
    for idx, (label, _, df) in enumerate(loaded):
        ax = axes[idx // ncols][idx % ncols]
        for boundary in PIECE_BOUNDARIES:
            ax.axvline(boundary + 0.5, color="0.8", linewidth=1, zorder=0)
        for name in names:
            g = df[df["benchmark"] == name].sort_values("span")
            if g.empty:
                continue
            color, linestyle = line_style(name)
            ax.plot(
                g["span"], g["display_value"],
                label=label_for_mode(name, args.mode),
                color=color, linestyle=linestyle, linewidth=1.5,
            )
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("Span of the mask")
        ax.set_ylabel(ylabel_for_unit(args.unit, suite))
        ax.set_xlim(1, df["span"].max())
        apply_yscale(ax, args.scale)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # One legend for all panels, as they share their variants and colors.
    handles = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2 * ncols)
    fig.suptitle(args.title or suite)

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale, args.sink)
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
