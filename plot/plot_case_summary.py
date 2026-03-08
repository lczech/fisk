#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os, sys

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


def _apply_order(grouped: pd.DataFrame, order: Optional[List[str]]) -> pd.DataFrame:
    if not order:
        return grouped

    present = grouped["benchmark"].tolist()
    order_present = [b for b in order if b in present]
    remaining = [b for b in present if b not in set(order_present)]
    final = order_present + remaining

    idx = pd.Index(final, name="benchmark")
    return grouped.set_index("benchmark").reindex(idx).reset_index()


def _colors_for_benchmarks(benchmarks: List[str], color_map: Dict[str, str]) -> List[str]:
    # Deterministic: use provided colors; for the rest, use Matplotlib's default cycle in order.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    colors: List[str] = []
    cycle_i = 0
    for b in benchmarks:
        if b in color_map:
            colors.append(color_map[b])
        else:
            colors.append(cycle[cycle_i % len(cycle)])
            cycle_i += 1
    return colors


def make_impl_summary_plot(
    df: pd.DataFrame,
    suite: Optional[str],
    title: str,
    outpath: Optional[str],
    benchmarks_keep: Optional[List[str]] = None,
    benchmark_order: Optional[List[str]] = None,
    benchmark_colors: Optional[Dict[str, str]] = None,
):
    # Filter for this suite (if given)
    if suite is not None:
        df = df[df["suite"] == suite]

    # Filter benchmarks (if given)
    if benchmarks_keep is not None:
        df_filtered = df[df["benchmark"].isin(benchmarks_keep)]

        # If filtering removed everything, fall back to full set
        if not df_filtered.empty:
            df = df_filtered

    # Aggregate per benchmark across all cases
    grouped = (
        df.groupby("benchmark")["ns_per_op"]
          .agg(["mean", "min", "max"])
          .reset_index()
    )

    # Reorder bars (if requested)
    grouped = _apply_order(grouped, benchmark_order)

    impls = grouped["benchmark"].tolist()
    means = grouped["mean"].values
    mins  = grouped["min"].values
    maxs  = grouped["max"].values

    # Asymmetric error bars: mean - min, max - mean
    lower_err = means - mins
    upper_err = maxs - means

    import numpy as np
    x = np.arange(len(impls))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = benchmark_colors or {}
    bar_colors = _colors_for_benchmarks(impls, cmap)

    ax.bar(x, means, width=width, label="mean", color=bar_colors)

    ax.errorbar(
        x,
        means,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=4,
        label="min/max",
    )

    # Add slanted mean labels, shifted right to avoid whiskers
    ymax = float(max(maxs)) if len(maxs) else 0.0
    x_offset = width * 0.18
    y_offset = ymax * 0.01 if ymax > 0 else 0.0

    for xi, yi in zip(x, means):
        ax.text(
            xi + x_offset,
            yi + y_offset,
            f"{yi:.2f}",
            ha="left",
            va="bottom",
            rotation=45,     # slight clockwise tilt
            fontsize=8,
        )

    # Extend y-axis slightly so labels stay inside the plot
    if ymax > 0:
        ax.set_ylim(top=ymax * 1.1)

    ax.set_title(title)
    # ax.set_xlabel("Implementation")
    ax.set_ylabel("Time per operation [ns]")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Wrote {outpath}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Per-implementation bar plot with min/max whiskers."
    )
    ap.add_argument("csv", help="Input results CSV (suite,case,benchmark,ns_per_op)")
    ap.add_argument(
        "--suite",
        default=None,
        help="Suite name to select (if omitted, use all suites together)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output image path (if omitted, show interactively)",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title override (default: suite name or generic)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    cpu = platform_from_csv_path(args.csv).replace("_", " ")
    suite = args.suite
    title = args.title or (suite if suite else cpu)

    make_impl_summary_plot(
        df=df,
        suite=suite,
        title=title,
        outpath=args.out,
        benchmarks_keep=BENCHMARKS_KEEP,
        benchmark_order=BENCHMARK_ORDER,
        benchmark_colors=BENCHMARK_COLORS,
    )


if __name__ == "__main__":
    main()
