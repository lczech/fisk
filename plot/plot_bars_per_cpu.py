#!/usr/bin/env python3
"""
Cross-platform benchmark summary plot.

Reads multiple benchmark CSV files (same schema: suite,case,benchmark,ns_per_op),
computes per-(benchmark, platform, compiler) mean over cases, and plots a grouped
bar chart:

- x-axis: benchmark (implementation name)
- for each benchmark: bars for each platform/compiler combination
- color: platform (CPU family / branding)
- hatch: compiler
- y-axis: mean ns/op across cases

Platform label is taken from the last path component of the directory containing the CSV.
Example:
    results/AMD EPYC 9684X, Clang 17/pext.csv
-> raw label "AMD EPYC 9684X, Clang 17"

The platform and compiler are then inferred by case-insensitive substring matching
against PLATFORM_ORDER and COMPILER_ORDER.

Usage examples:
  python plot_bars_per_cpu.py \
      --file "results/AMD EPYC 9684X, Clang 17/pext.csv" \
      --file "results/AMD EPYC 9684X, GCC 13/pext.csv" \
      --suite PEXT \
      --out pext_platforms.png

  python plot_bars_per_cpu.py --glob "artifacts/*/pext.csv" --suite PEXT

If --out is omitted, shows interactively.
"""

from __future__ import annotations

import argparse
import glob
import os, sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import matplotlib as mpl
import numpy as np

# hard coded line with for this case
mpl.rcParams["hatch.linewidth"] = 0.7


# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *

PLATFORM_ORDER = [
    "Epyc",
    "Ryzen",
    "Xeon",
    "M1",
    "M2",
    "M3",
]

COMPILER_ORDER = [
    "Clang",
    "GCC",
]

TITLE_FROM_FILENAME = {
    "bit_extract_weights.csv": "Bit extract with different mask weights",
    "bit_extract_blocks.csv":  "Bit extract with different block sizes in the mask",
    "seq_enc.csv":             "Sequence encoding from char to two-bit codes",
    "kmer_extract.csv":        "Extraction from sequence to two-bit coded k-mers",
    "kmer_spaced_multi.csv":   "Extraction from sequence to spaced k-mers with multiple masks",
    "kmer_spaced_single.csv":  "Extraction from sequence to spaced k-mers with a single mask",
    "kmer_clark.csv":          "K-mer Clark test",
}


def raw_label_from_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    return p.parent.name


def infer_platform(raw_label: str) -> str:
    s = raw_label.casefold()
    for plat in PLATFORM_ORDER:
        if plat.casefold() in s:
            return plat
    return "Other"


def infer_compiler(raw_label: str) -> str:
    s = raw_label.casefold()
    for comp in COMPILER_ORDER:
        if comp.casefold() in s:
            return comp
    return "Other"


def load_one(
    csv_path: str,
    raw_label: str | None = None,
    plot_benchmarks: set[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if raw_label is None:
        raw_label = raw_label_from_csv_path(csv_path)

    platform = infer_platform(raw_label)
    compiler = infer_compiler(raw_label)

    df["raw_label"] = raw_label
    df["platform"] = platform
    df["compiler"] = compiler
    df["source"] = str(csv_path)

    if plot_benchmarks is not None:
        df = df[df["benchmark"].isin(plot_benchmarks)]

    return df


def color_for_platform(platform: str) -> str:
    platform_colors = {
        "Epyc":  "#08aa72",  # green
        "Ryzen": "#fb6516",  # orange
        "Xeon":  "#0874d1",  # blue
        "M1":    "#979797",  # gray
        "M2":    "#979797",  # gray
        "M3":    "#979797",  # gray
        "Other": "#b0b0b0",  # fallback
    }
    return platform_colors.get(platform, "grey")


def hatch_for_compiler(compiler: str) -> str:
    compiler_hatches = {
        "Clang": "",
        "GCC": "///",
        "Other": "xx",
    }
    return compiler_hatches.get(compiler, "xx")


def series_sort_key(series_name: str) -> tuple[int, int, str]:
    """
    Series name format: '<platform> | <compiler>'
    """
    if " | " in series_name:
        platform, compiler = series_name.split(" | ", 1)
    else:
        platform, compiler = series_name, "Other"

    try:
        p_idx = PLATFORM_ORDER.index(platform)
    except ValueError:
        p_idx = len(PLATFORM_ORDER)

    try:
        c_idx = COMPILER_ORDER.index(compiler)
    except ValueError:
        c_idx = len(COMPILER_ORDER)

    return (p_idx, c_idx, series_name)


def legend_label(series_name: str) -> str:
    platform, compiler = series_name.split(" | ", 1)
    return f"{platform} / {compiler}"


def main() -> None:

    # --------------------------------------
    #      CLI args
    # --------------------------------------

    ap = argparse.ArgumentParser(description="Cross-platform grouped bar plot (mean over cases).")
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        help="CSV file path. Can be given multiple times.",
    )
    ap.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern(s) for CSV files, e.g. 'artifacts/*/pext.csv'. Can be given multiple times.",
    )
    ap.add_argument(
        "--suite",
        default=None,
        help="Suite name to filter (recommended). If omitted, all suites are combined.",
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Use BENCHMARKS_KEEP_EXTENDED instead of BENCHMARKS_KEEP",
    )
    ap.add_argument(
        "--reduced",
        action="store_true",
        help="Use BENCHMARKS_KEEP_REDUCED instead of BENCHMARKS_KEEP",
    )
    ap.add_argument(
        "--untight",
        action="store_true",
        help="Use BENCHMARKS_KEEP_REDUCED instead of BENCHMARKS_KEEP",
    )
    ap.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide legend in the plot.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title override.",
    )
    ap.add_argument(
        "--y-lim",
        type=float,
        help="Y-axis max limit.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure (default: 300).",
    )
    args = ap.parse_args()

    # --------------------------------------
    #      Load the data
    # --------------------------------------

    benchmarks_keep = BENCHMARKS_KEEP_EXTENDED if args.extended else BENCHMARKS_KEEP
    benchmarks_keep = BENCHMARKS_KEEP_REDUCED if args.reduced else benchmarks_keep
    plot_benchmarks = set(benchmarks_keep)

    files: list[str] = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))

    files = [f for f in files if f]
    if not files:
        raise SystemExit("No input files. Provide --file ... or --glob ...")

    dfs = [load_one(f, plot_benchmarks=plot_benchmarks) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    required_cols = {"suite", "case", "benchmark", "ns_per_op", "platform", "compiler"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in input CSV(s): {sorted(missing)}")

    suite = args.suite
    if suite is not None:
        df = df[df["suite"] == suite]
        if df.empty:
            available = sorted(set(pd.concat(dfs)["suite"].astype(str)))
            raise SystemExit(f"No rows for suite='{suite}'. Available suites: {available}")

    # Aggregate: mean over cases for each (benchmark, platform, compiler)
    agg = (
        df.groupby(["benchmark", "platform", "compiler"], as_index=False)["ns_per_op"]
          .mean()
          .rename(columns={"ns_per_op": "mean_ns_per_op"})
    )

    agg["series"] = agg["platform"] + " | " + agg["compiler"]

    # Pivot for plotting: rows=benchmark, columns=series
    pivot = agg.pivot(index="benchmark", columns="series", values="mean_ns_per_op")

    # Stable benchmark order
    benchmark_order = [b for b in BENCHMARK_ORDER if b in pivot.index]
    benchmark_order += [b for b in pivot.index if b not in benchmark_order]
    pivot = pivot.reindex(benchmark_order)
    benchmarks = pivot.index.tolist()

    # Stable series order: platform first, then compiler
    series = sorted(list(pivot.columns), key=series_sort_key)
    pivot = pivot.reindex(columns=series)
    series = pivot.columns.tolist()

    # --------------------------------------
    #      Generate the plot
    # --------------------------------------

    if args.y_lim is not None:
        YLIM = args.y_lim
    else:
        YLIM = 10.0
    YMAX = 0.88 * YLIM
    if YLIM > 20:
        YMAX = YLIM

    x = np.arange(len(benchmarks))
    group_width = 0.8
    bar_w = group_width / max(1, len(series))

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(benchmarks) + 4), 6))

    # for j, s in enumerate(series):
    #     vals = pivot[s].values
    #     platform, compiler = s.split(" | ", 1)

    #     ax.bar(
    #         x + (j - (len(series) - 1) / 2.0) * bar_w,
    #         vals,
    #         width=bar_w,
    #         label=legend_label(s),
    #         color=color_for_platform(platform),
    #         hatch=hatch_for_compiler(compiler),
    #         edgecolor="black",
    #         linewidth=0.8,
    #     )

    # We want to print the minimum per platform, as that's the fastest algorithm.
    best_impl = []

    # Plot all bars, manually for full control
    for j, s in enumerate(series):
        vals = pivot[s].values
        platform, compiler = s.split(" | ", 1)

        # Find the minimum for this platform
        # print(legend_label(s), str(vals))
        min_idx = np.nanargmin(vals)
        best_impl.append([legend_label(s), min_idx, vals[min_idx]])

        xpos = x + (j - (len(series) - 1) / 2.0) * bar_w
        plot_vals = [min(v, YMAX) for v in vals]

        bars = ax.bar(
            xpos,
            plot_vals,
            width=bar_w,
            label=legend_label(s),
            color=color_for_platform(platform),
            hatch=hatch_for_compiler(compiler),
            edgecolor="white",
            linewidth=0.0,
        )

        # Mark truncated bars and annotate true values
        for xi, shown_v, true_v, bar in zip(xpos, plot_vals, vals, bars):
            if pd.notna(true_v) and true_v > YMAX:
                x_left = bar.get_x()
                x_right = x_left + bar.get_width()
                width = x_right - x_left

                # Place the cut near the top of the visible bar
                # y_bottom_left = YMAX - 0.92
                # y_top_left    = YMAX - 0.62
                y_bottom_left = YMAX - 0.2
                y_top_left    = YMAX
                slope = 0.16

                # Make the white patch slightly wider so it visually reaches the bar outline
                overhang = width * 0.01
                xl = x_left - overhang
                xr = x_right + overhang

                # Three zig-zags across the width
                n_zigs = 3
                step = (xr - xl) / n_zigs
                amp = 0.07  # zig-zag amplitude in y-direction

                def zigzag_points(y_left):
                    pts = []
                    for i in range(n_zigs):
                        x0 = xl + i * step
                        x1 = x0 + step / 2.0
                        x2 = x0 + step
                        y0 = y_left + slope * ((x0 - xl) / (xr - xl))
                        y1 = y_left + slope * ((x1 - xl) / (xr - xl)) + amp
                        y2 = y_left + slope * ((x2 - xl) / (xr - xl))
                        if i == 0:
                            pts.append((x0, y0))
                        pts.append((x1, y1))
                        pts.append((x2, y2))
                    return pts

                bottom_edge = zigzag_points(y_bottom_left)
                top_edge = zigzag_points(y_top_left)

                # Polygon: bottom zig-zag left->right, then top zig-zag right->left
                cut_poly_pts = bottom_edge + list(reversed(top_edge))

                cut_patch = plt.Polygon(
                    cut_poly_pts,
                    closed=True,
                    facecolor="white",
                    edgecolor="none",
                    zorder=5,
                    clip_on=True,
                )
                cut_patch.set_clip_path(bar)
                ax.add_patch(cut_patch)

                # # Draw the zig-zag boundaries on top
                # line_bottom, = ax.plot(
                #     [p[0] for p in bottom_edge],
                #     [p[1] for p in bottom_edge],
                #     color="black",
                #     linewidth=0.8,
                #     zorder=6,
                # )
                # line_top, = ax.plot(
                #     [p[0] for p in top_edge],
                #     [p[1] for p in top_edge],
                #     color="black",
                #     linewidth=0.8,
                #     zorder=6,
                # )
                # line_bottom.set_clip_path(bar)
                # line_top.set_clip_path(bar)

                # Smaller label with the true value
                label = f"{true_v:.0f}" if float(true_v).is_integer() else f"{true_v:.1f}"
                ax.text(
                    xi,
                    YMAX + 0.10,
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=10,
                    clip_on=False,
                )

    # --------------------------------------
    #      Axes and finalization
    # --------------------------------------

    title = args.title
    if title is None:
        first_file = Path(files[0]).name
        title = TITLE_FROM_FILENAME.get(
            first_file,
            f"{suite} cross-platform summary" if suite else "Cross-platform summary",
        )

    ax.set_title(title)
    ax.set_ylabel("Time per operation [ns]")
    ax.set_xticks(x)
    labels = [BENCHMARK_RENAMES.get(b, b) for b in benchmarks]
    if args.reduced:
        labels = [BENCHMARK_RENAMES_REDUCED.get(b, b) for b in benchmarks]

    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(0, YLIM)
    if YLIM <= 20:
        ax.set_yticks(range(0, int(round(YMAX)) + 1, 1))

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, leg_labels = ax.get_legend_handles_labels()
    if not args.no_legend:
        ax.legend(
            handles,
            leg_labels,
            title="CPU / Compiler",
            loc="upper right",
            ncol=2 if len(leg_labels) > 5 else 3
        )

    if args.untight:
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.94)
    else:
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    # ax.margins(x=0)
    if len(labels) > 0:
        ax.set_xlim(-0.5, len(benchmarks) - 0.5)

    if args.out:
        out = args.out

        # print best implementation for each arch
        # print(title)
        root, ext = os.path.splitext(out)
        with open(f"{root}.csv", 'w') as f:
            f.write(f"platform,best_impl,best_time\n")
            for impl in best_impl:
                f.write(f"{impl[0]},{labels[impl[1]]},{impl[2]}\n")
                # name = impl[0]
                # min_val = impl[1]
                # min_idx = impl[2]
                # print(name, min_idx, labels[min_idx], min_val)

        # Append suffix if extended mode is used
        # if args.extended and out is not None:
        #     root, ext = os.path.splitext(out)
        #     out = f"{root}_ext{ext}"

        parent = str(Path(out).parent)
        if parent not in ("", "."):
            os.makedirs(parent, exist_ok=True)

        fig.savefig(out, dpi=args.dpi)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
