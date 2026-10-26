#!/usr/bin/env python3
"""
K-mer extraction: one line per variant over k, for one CPU/compiler.

Reads the tables of one or more contiguous k-mer suites of one CPU/compiler directory, such as
kmer_extract_ascii_checked.csv and kmer_extract_ascii_unchecked.csv, plotted together. By default,
one panel shows the rows of one encoding and layout (--encoding, --layout); with --layout-panels,
one panel per layout shows all rows of that layout, with a shared legend below them, as for the
extraction from packed sequences.

Usage examples:
  python plot/plot_kmer_extract.py \
      "results/AMD EPYC 9684X, Clang 17/kmer_extract_ascii_checked.csv" \
      "results/AMD EPYC 9684X, Clang 17/kmer_extract_ascii_unchecked.csv" \
      --mode supp --out kmer_extract_ascii.png

  python plot/plot_kmer_extract.py --layout-panels --mode main \
      "results/AMD EPYC 9684X, Clang 17/kmer_extract_packed_unchecked.csv"

If --out is omitted, shows interactively.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
import os, sys

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


# Fallback palette for any variant name not (yet) listed in BENCHMARK_COLORS in the layout panels
# -- e.g. a brand new experimental variant added before its color is curated there. Assigned
# deterministically by position, so a given uncurated name still gets a stable color within one
# run at least.
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


def _plot_single_panel(df, plot_order, cpu, suite, args):
    """One panel of the rows of one encoding and layout, with the legend inside."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in plot_order:
        g = df[df["benchmark"] == name].sort_values("k")

        color = BENCHMARK_COLORS.get(name)
        linestyle = BENCHMARK_LINESTYLES.get(name)

        plot_kwargs = {
            "marker": ".",
            "linewidth": 2,
            "label": benchmark_label(name),
        }

        if color is not None:
            plot_kwargs["color"] = color
        if linestyle is not None:
            plot_kwargs["linestyle"] = linestyle

        ax.plot(g["k"], g["display_value"], **plot_kwargs)

    ax.set_xlabel("k-mer size (k)")
    ax.set_ylabel(ylabel_for_unit(args.unit, suite))
    ax.set_title(args.title or cpu.replace("_", " "))

    ax.set_xlim(1, 32)
    apply_yscale(ax, args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale)
    if args.y_min is not None:
        y_min = args.y_min
    if args.y_max is not None:
        y_max = args.y_max
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", handlelength=2.75)

    fig.tight_layout()
    return fig


def _plot_layout_panel(ax, df, layout_value, plot_order, colors, linestyle_map):
    g_layout = df[df["layout"] == layout_value]
    for name in plot_order:
        g = g_layout[g_layout["benchmark"] == name].sort_values("k")
        if g.empty:
            continue
        ax.plot(
            g["k"],
            g["display_value"],
            marker=".",
            linewidth=2,
            color=colors[name],
            linestyle=linestyle_map.get(name),
        )
    ax.set_title(layout_value)
    ax.set_xlabel("k-mer size (k)")
    ax.set_xlim(1, 32)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def _plot_layout_panels(df, plot_order, cpu, suite, args):
    """One panel per layout, sharing their y-axis, with one legend for both below them."""
    colors = _colors_for_benchmarks(plot_order, BENCHMARK_COLORS)
    fig, (ax_msb, ax_lsb) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _plot_layout_panel(ax_msb, df, "msb", plot_order, colors, BENCHMARK_LINESTYLES)
    _plot_layout_panel(ax_lsb, df, "lsb", plot_order, colors, BENCHMARK_LINESTYLES)

    ax_msb.set_ylabel(ylabel_for_unit(args.unit, suite))
    apply_yscale(ax_msb, args.scale)
    apply_yscale(ax_lsb, args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale)
    if args.y_min is not None:
        y_min = args.y_min
    if args.y_max is not None:
        y_max = args.y_max
    ax_msb.set_ylim(y_min, y_max)

    fig.suptitle(args.title or cpu.replace("_", " "))

    # Single shared legend for the whole figure: colors/labels are identical across
    # both panels, so build it once from the plot order rather than duplicating a
    # legend per subplot.
    handles = [
        Line2D([], [], color=colors[name], linestyle=BENCHMARK_LINESTYLES.get(name), linewidth=2)
        for name in plot_order
    ]
    labels = [benchmark_label(name) for name in plot_order]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), handlelength=2.75)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot k-mer extract benchmark results")
    parser.add_argument(
        "csv", nargs="+",
        help="CSV files of the contiguous k-mer suites of one CPU/compiler, plotted together; "
             "missing ones are left out",
    )
    parser.add_argument("--title", default=None,
                        help="Plot title (default: platform name from the CSV's directory)")
    parser.add_argument("--out", default=None,
                        help="Output image file (e.g. kmer_extract_ascii.png). If omitted, show "
                             "interactively.")
    parser.add_argument("--layout-panels", action="store_true",
                        help="One panel per layout, with all rows of that layout, instead of one "
                             "panel of the rows of --encoding and --layout")
    parser.add_argument("--encoding", default="acgt",
                        help="Which encoding's rows to plot (default: acgt, the only encoding that "
                             "all variants are run with)")
    parser.add_argument("--layout", default="msb",
                        help="Which layout's rows to plot (default: msb, the only layout that all "
                             "variants are run with)")
    add_mode_arg(parser)
    add_unit_scale_args(parser)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    csvs = [p for p in args.csv if os.path.exists(p)]
    for p in args.csv:
        if p not in csvs:
            warn(f"{p} missing, leaving it out of the plot")
    if not csvs:
        raise SystemExit("None of the input CSV files exist")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    cpu = platform_from_csv_path(csvs[0])
    suite = single_suite(df)

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # "case" looks like "encoding=acgt;layout=msb;k=17" -- split into one column per field.
    df = parse_case_fields(df)
    df["k"] = df["k"].astype(int)
    # Older result files have plain "k=17" or "layout=msb;k=17" cases; rows without a field are
    # ACGT and MSB.
    for field, default in [("encoding", "acgt"), ("layout", "msb")]:
        if field not in df.columns:
            df[field] = default
        df[field] = df[field].fillna(default)
    if not args.layout_panels:
        df = df[(df["encoding"] == args.encoding) & (df["layout"] == args.layout)]

    df = filter_mode(df, args.mode)
    if df.empty:
        raise ValueError(f"No rows left after filtering for mode {args.mode} in {csvs!r}")

    # Convert to the requested display unit once, up front -- no aggregation happens in this
    # script (each row is already one (benchmark, layout, k) point), so there's no ordering
    # concern between conversion and aggregation here.
    df["display_value"] = convert_for_display(df["ns_per_op"], args.unit)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    present = df["benchmark"].drop_duplicates().tolist()
    plot_order = _ordered_benchmarks(present, BENCHMARK_ORDER)
    if args.layout_panels:
        fig = _plot_layout_panels(df, plot_order, cpu, suite, args)
    else:
        fig = _plot_single_panel(df, plot_order, cpu, suite, args)

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    if out:
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
