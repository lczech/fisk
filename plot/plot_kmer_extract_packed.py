#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
import sys

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


# Fallback palette for any variant name not (yet) listed in PACKED_KMER_VARIANT_COLORS -- e.g. a
# brand new experimental variant added before its color is curated there. Assigned deterministically
# by position, so a given uncurated name still gets a stable color within one run at least.
FALLBACK_PALETTE = plt.get_cmap("tab20").colors


def _ordered_benchmarks(present: List[str], order: Optional[List[str]]) -> List[str]:
    """
    Return benchmarks in the desired plotting order:
    - first, those in 'order' that are present
    - then, any remaining benchmarks in their current (first-appearance) order
    """
    if not order:
        return present
    order_present = [b for b in order if b in present]
    remaining = [b for b in present if b not in set(order_present)]
    return order_present + remaining


def _colors_for_benchmarks(names: List[str], color_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build a name -> color dict, once, for a fixed list of benchmark names: names
    already in `color_map` keep that color, everything else gets a color from
    FALLBACK_PALETTE by position. Building this once (rather than letting each
    subplot's Axes auto-cycle its own colors) is what keeps a given benchmark's
    color identical across the msb and lsb panels.
    """
    colors = {}
    for i, name in enumerate(names):
        colors[name] = color_map.get(name, FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])
    return colors


def _label_for_benchmark(name: str, rename_map: Dict[str, str]) -> str:
    return rename_map.get(name, name)


def _plot_panel(ax, df, order_value, plot_order, colors, linestyle_map):
    g_order = df[df["order"] == order_value]
    for name in plot_order:
        g = g_order[g_order["benchmark"] == name].sort_values("k")
        if g.empty:
            continue
        ax.plot(
            g["k"],
            g["ns_per_op"],
            marker=".",
            linewidth=2,
            color=colors[name],
            linestyle=linestyle_map.get(name),
        )
    ax.set_title(order_value)
    ax.set_xlabel("k-mer size (k)")
    ax.set_xlim(1, 32)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot packed k-mer extract benchmark results")
    parser.add_argument("csv", help="CSV file produced by bench_kmer_extract_packed")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: platform name from the CSV's directory)")
    parser.add_argument("--out", default=None,
                        help="Output image file (e.g. kmer_extract_packed.png). If omitted, show interactively.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    df = pd.read_csv(args.csv)
    cpu = platform_from_csv_path(args.csv)

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # "case" looks like "order=msb;k=17" -- split into an "order" and a "k" column.
    df = parse_case_fields(df)
    df["k"] = df["k"].astype(int)

    if df.empty:
        raise ValueError(f"No rows found in CSV file {args.csv!r}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    # Plot every benchmark present -- no BENCHMARKS_KEEP filtering here.
    present = df["benchmark"].drop_duplicates().tolist()
    plot_order = _ordered_benchmarks(present, PACKED_KMER_VARIANT_ORDER)
    colors = _colors_for_benchmarks(plot_order, PACKED_KMER_VARIANT_COLORS)

    fig, (ax_msb, ax_lsb) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _plot_panel(ax_msb, df, "msb", plot_order, colors, BENCHMARK_LINESTYLES)
    _plot_panel(ax_lsb, df, "lsb", plot_order, colors, BENCHMARK_LINESTYLES)

    ax_msb.set_ylabel("Time per operation [ns]")
    ymax = float(df["ns_per_op"].max())
    ax_msb.set_ylim(0, ymax * 1.05)

    fig.suptitle(args.title or cpu.replace("_", " "))

    # Single shared legend for the whole figure: colors/labels are identical across
    # both panels, so build it once from the plot order rather than duplicating a
    # legend per subplot.
    handles = [
        Line2D([], [], color=colors[name], linestyle=BENCHMARK_LINESTYLES.get(name), linewidth=2)
        for name in plot_order
    ]
    labels = [_label_for_benchmark(name, BENCHMARK_RENAMES) for name in plot_order]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), handlelength=2.75)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    if args.out:
        fig.savefig(args.out, dpi=300)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
