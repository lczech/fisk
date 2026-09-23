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


def make_grouped_bar_plot_impl_first(
    df, suite, title, outpath,
    unit: str = "ops", scale: str = "log",
    y_min: float | None = None, y_max: float | None = None,
):
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
    #
    # Aggregation (the "mean" below, only relevant if there are duplicate rows) happens in ns/op
    # space, and only the resulting pivot table is converted to the display unit afterwards --
    # converting per-row first and averaging throughput values would be a different, non-equivalent
    # statistic (see convert_for_display()'s docstring).
    pivot = df.pivot_table(
        index="benchmark",
        columns="case",
        values="ns_per_op",
        aggfunc="mean",  # in case there are duplicates
    )
    pivot = convert_for_display(pivot, unit)

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
    apply_yscale(ax, scale)
    axis_y_min, axis_y_max = compute_axis_limits(pivot.values, scale)
    # Extra top headroom beyond compute_axis_limits()'s default, reserved for the rotated
    # value-label text drawn above every bar below -- on a compressed "log" axis a fixed 5%
    # doesn't leave enough room for the label's full glyph height, so it needs proportionally
    # more than the plain axis padding used by scripts that don't draw a label per bar.
    axis_y_max *= 1.3 if scale == "log" else 1.1
    if y_min is not None:
        axis_y_min = y_min
    if y_max is not None:
        axis_y_max = y_max

    for j, case in enumerate(cases):
        vals = pivot[case].values
        xs = x + offsets[j]

        # colors = [BENCHMARK_COLORS[bench] for bench in impls]
        # bars = ax.bar(xs, vals, width=bar_width, label=str(case), color=colors)
        bars = ax.bar(xs, vals, width=bar_width, label=str(case), color=case_colors[j])

        # Add value labels above each bar. Multiplicative headroom on "log" (an additive offset
        # would be disproportionately large next to a small bar on a log axis); additive on
        # "linear", sized off the axis span rather than the data max so it stays sensible even
        # when --y-max widens the axis well past the tallest bar.
        for bar, val in zip(bars, vals):
            if pd.notna(val):
                label_y = val * 1.02 if scale == "log" else bar.get_height() + (axis_y_max - axis_y_min) * 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=10,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel_for_unit(unit, suite))
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")

    # y-limit chosen to cover all values in this plot consistently; see compute_axis_limits().
    ax.set_ylim(axis_y_min, axis_y_max)

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # fig.tight_layout()
    #
    # left=0.08 was tuned for "ns" tick labels (short, e.g. "10", "50"); "ops" throughput values
    # are typically sub-1 decimals (e.g. "0.05", "0.2"), whose wider tick labels need more room or
    # the y-axis label gets clipped off the left edge of the figure.
    left = 0.095 if unit == "ops" else 0.08
    fig.subplots_adjust(left=left, right=0.99, bottom=0.30, top=0.90)

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
    add_unit_scale_args(ap)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    cpu = platform_from_csv_path(args.csv).replace("_", " ")
    suite = args.suite
    title = args.title or (suite if suite else cpu)

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    make_grouped_bar_plot_impl_first(
        df, suite, title, out,
        unit=args.unit, scale=args.scale, y_min=args.y_min, y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
