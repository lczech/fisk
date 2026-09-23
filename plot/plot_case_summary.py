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


def _label_for_benchmark(name: str, rename_map: Dict[str, str]) -> str:
    return rename_map.get(name, name)


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
    unit: str = "ops",
    scale: str = "log",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
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

    # Aggregate per benchmark across all cases, in ns/op space -- converting to the display unit
    # happens below, on the aggregated scalars only. Converting each row to throughput first and
    # then averaging would be a different, non-equivalent statistic (see convert_for_display()'s
    # docstring).
    grouped = (
        df.groupby("benchmark")["ns_per_op"]
          .agg(["mean", "min", "max"])
          .reset_index()
    )

    # Reorder bars (if requested)
    grouped = _apply_order(grouped, benchmark_order)

    import numpy as np

    impls = grouped["benchmark"].tolist()
    means_ns = grouped["mean"].values
    mins_ns  = grouped["min"].values
    maxs_ns  = grouped["max"].values

    means = convert_for_display(means_ns, unit)
    # Throughput inverts the ordering: the fastest (min-time) measurement becomes the *highest*
    # throughput, and vice versa -- so which raw bound feeds the lower/upper whisker has to swap
    # along with the unit.
    lo_ns, hi_ns = (mins_ns, maxs_ns) if unit == "ns" else (maxs_ns, mins_ns)
    lo_disp = convert_for_display(lo_ns, unit)
    hi_disp = convert_for_display(hi_ns, unit)

    # Asymmetric error bars: mean - lower bound, upper bound - mean
    lower_err = means - lo_disp
    upper_err = hi_disp - means

    x = np.arange(len(impls))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 8))
    apply_yscale(ax, scale)

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

    axis_y_min, axis_y_max = compute_axis_limits(
        np.concatenate([means, lo_disp, hi_disp]), scale
    )
    # Extra top headroom beyond compute_axis_limits()'s default, reserved for the rotated
    # value-label text drawn above every bar below -- on a compressed "log" axis a fixed 5%
    # doesn't leave enough room for the label's full glyph height, so it needs proportionally
    # more than the plain axis padding used by scripts that don't draw a label per bar.
    axis_y_max *= 1.3 if scale == "log" else 1.1
    if y_min is not None:
        axis_y_min = y_min
    if y_max is not None:
        axis_y_max = y_max

    # Add slanted mean labels, shifted right to avoid whiskers. Multiplicative headroom on "log"
    # (an additive offset would be disproportionately large next to a small bar on a log axis);
    # additive on "linear", sized off the axis span.
    x_offset = width * 0.18

    for xi, yi in zip(x, means):
        label_y = yi * 1.02 if scale == "log" else yi + (axis_y_max - axis_y_min) * 0.015
        ax.text(
            xi + x_offset,
            label_y,
            f"{yi:.2f}",
            ha="left",
            va="bottom",
            rotation=45,     # slight clockwise tilt
            fontsize=12,
        )

    ax.set_title(title)
    # ax.set_xlabel("Implementation")
    ax.set_ylabel(ylabel_for_unit(unit, suite))
    ax.set_xticks(x)
    # ax.set_xticklabels(impls, rotation=45, ha="right")
    labels = [_label_for_benchmark(name, BENCHMARK_RENAMES) for name in impls]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # y-limit chosen to cover all values in this plot consistently; see compute_axis_limits().
    ax.set_ylim(axis_y_min, axis_y_max)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    # Fix layout such that white space does not differ if we have inputs here
    # with different benchmarks. Tight layout would otherwise cause the bottom axis
    # label to take up a different amount (tight) of whitespace across plots,
    # making them misaligned when put next to each other in the manusript.
    # fig.tight_layout()
    #
    # left=0.06 was tuned for "ns" tick labels (short, e.g. "10", "50"); "ops" throughput values
    # are typically sub-1 decimals (e.g. "0.05", "0.2"), whose wider tick labels need more room or
    # the y-axis label gets clipped off the left edge of the figure.
    left = 0.085 if unit == "ops" else 0.06
    fig.subplots_adjust(left=left, right=0.99, bottom=0.30, top=0.96)

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
        "--extended",
        action="store_true",
        help="Use extended benchmark set (BENCHMARKS_KEEP_EXTENDED) and append _ext to output filename",
    )
    ap.add_argument(
        "--reduced",
        action="store_true",
        help="Use BENCHMARKS_KEEP_REDUCED instead of BENCHMARKS_KEEP",
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

    outpath = args.out
    if args.extended and outpath is not None:
        root, ext = os.path.splitext(outpath)
        outpath = f"{root}_ext{ext}"
    if args.reduced and outpath is not None:
        root, ext = os.path.splitext(outpath)
        outpath = f"{root}_red{ext}"
    outpath = apply_unit_scale_suffix(outpath, args.unit, args.scale)

    benchmarks_keep = BENCHMARKS_KEEP_EXTENDED if args.extended else BENCHMARKS_KEEP
    benchmarks_keep = BENCHMARKS_KEEP_REDUCED if args.reduced else benchmarks_keep

    make_impl_summary_plot(
        df=df,
        suite=suite,
        title=title,
        outpath=outpath,
        benchmarks_keep=benchmarks_keep,
        benchmark_order=BENCHMARK_ORDER,
        benchmark_colors=BENCHMARK_COLORS,
        unit=args.unit,
        scale=args.scale,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
