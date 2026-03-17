#!/usr/bin/env python3
import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


def make_grouped_bar_plot_impl_first(df, suite, title, outpath):
    # Filter for this suite (if given)
    if suite is not None:
        df = df[df["suite"] == suite]

    if df.empty:
        raise ValueError(f"No data found for suite: {suite}")

    # Keep only selected benchmarks
    df = df[df["benchmark"].isin(BENCHMARKS_KEEP)]

    if df.empty:
        raise ValueError("No data left after filtering with BENCHMARKS_KEEP")

    # Pivot so index = implementation (benchmark), columns = case
    # Each row will be one implementation; columns are the cases
    pivot = df.pivot_table(
        index="benchmark",
        columns="case",
        values="ns_per_op",
        aggfunc="mean",  # in case there are duplicates
    )

    # Apply benchmark order, keeping only those actually present
    ordered_impls = [b for b in BENCHMARK_ORDER if b in pivot.index]
    if not ordered_impls:
        raise ValueError("None of the BENCHMARK_ORDER entries are present in the data")

    pivot = pivot.reindex(ordered_impls)

    impls = list(pivot.index)    # implementations on x-axis
    cases = list(pivot.columns)  # one bar per case within each impl group

    num_impls = len(impls)
    num_cases = len(cases)

    # Use viridis but trim extreme ends and reverse
    cmap = plt.colormaps["viridis_r"]
    lo = 0.1   # avoid very dark end
    hi = 0.9   # avoid very bright end
    case_colors = [
        cmap(lo + (hi - lo) * (i / max(1, num_cases - 1)))
        for i in range(num_cases)
]

    # X positions: one group per implementation
    x = np.arange(num_impls)

    # Width of each bar inside a group
    group_width = 0.8
    bar_width = group_width / max(1, num_cases)

    # Center the bars within each group
    offsets = [
        (j - (num_cases - 1) / 2.0) * bar_width
        for j in range(num_cases)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ymax = float(pivot.max().max())

    for j, case in enumerate(cases):
        vals = pivot[case].values
        xs = x + offsets[j]

        # colors = [BENCHMARK_COLORS[bench] for bench in impls]
        # bars = ax.bar(xs, vals, width=bar_width, label=str(case), color=colors)
        bars = ax.bar(xs, vals, width=bar_width, label=str(case), color=case_colors[j])

        # Add value labels above each bar
        for bar, val in zip(bars, vals):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + ymax * 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=10,
                )

    ax.set_title(title)
    ax.set_ylabel("Time per operation [ns]")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")

    # set y-limit such that it coveres all values we have consistently
    ax.set_ylim(0, 55)

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.90)

    if outpath:
        fig.savefig(outpath, dpi=300)
        print(f"Wrote {outpath}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Grouped bar plot: all cases per implementation."
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

    make_grouped_bar_plot_impl_first(df, suite, title, args.out)


if __name__ == "__main__":
    main()
