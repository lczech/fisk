#!/usr/bin/env python3
"""
K-mer extraction: the variants of one suite across CPUs/compilers, at a few cases each.

Reads the tables of one k-mer suite, or of the checked and unchecked suite of a pair, of one or
more CPU/compiler directories, and draws per case point a grouped bar chart (one group per
platform/compiler, one bar per variant), optionally a heatmap relative to the fastest variant of
each platform, and one summary of the best variant per platform. The case points are the
combinations of the case fields, such as the layout, with k at the values of --k, by default the
two tiers of the extraction from packed sequences, 29 (narrow-width) and 32 (full-width), and every
mask of the spaced suites.

Usage example:
  python plot/plot_kmer_extract_compare_per_cpu.py \
      --glob "results/*/kmer_extract_packed_unchecked.csv" \
      --name kmer_extract_packed_unchecked --out-dir results/Summaries
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '.')
from plot_common import (
    BENCHMARK_COLORS,
    BENCHMARK_ORDER,
    add_mode_arg,
    filter_mode,
    label_for_mode,
    parse_case_fields,
    platform_compiler_sort_key,
    platform_from_csv_path,
    warn,
)

# Order and colors come from plot_common.py, shared with the other plots, so a given variant's
# color is identical across every plot in this whole family, not just within this file. Variants
# without a curated color get one from FALLBACK_PALETTE, by position.
FALLBACK_PALETTE = plt.get_cmap("tab20").colors

# The default values of k of the case points: the two tiers of the extraction from packed
# sequences.
DEFAULT_KS = [29, 32]


def load_all(csv_paths: List[str], ks: List[int]) -> pd.DataFrame:
    """
    Load each given kmer_extract_packed_unchecked.csv, tag each row with its platform (the parent
    directory name, e.g. "AMD EPYC 9684X, Clang 17"), and concatenate into one long-format
    DataFrame. Takes
    an explicit file list (built by the caller, e.g. from plot_all_cpus.sh's curated CPUS array via
    --file, same as plot_bars_per_cpu.py) rather than globbing a results directory itself, so the
    set of platforms shown here always matches the rest of the cross-CPU plots -- not a second,
    independently-curated selection. Keeps the rows with k in `ks`.
    """
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["platform"] = platform_from_csv_path(csv_path)
        frames.append(df)
    if not frames:
        raise ValueError("No input files. Provide --file ... or --glob ...")
    df = parse_case_fields(pd.concat(frames, ignore_index=True))
    if "k" in df.columns:
        df = df[df["k"].astype(int).isin(ks)].copy()
    return df


def case_points(df: pd.DataFrame) -> List[tuple]:
    """The case points: the combinations of the case fields present in the rows, in order."""
    fields = point_fields(df)
    points = df[fields].drop_duplicates()
    return sorted(points.itertuples(index=False, name=None), key=_point_sort_key)


def point_fields(df: pd.DataFrame) -> List[str]:
    """The case fields of the rows, in the order of the case strings."""
    return [f for f in _case_keys(df) if f in df.columns]


def _case_keys(df: pd.DataFrame) -> List[str]:
    keys = []
    for case in df["case"].drop_duplicates():
        for part in case.split(";"):
            key = part.split("=", 1)[0]
            if key not in keys:
                keys.append(key)
    return keys


def _point_sort_key(point):
    # Numbers by value, so that k=29 comes before k=32, and mask=2 before mask=10.
    return tuple((0, int(v), "") if str(v).isdigit() else (1, 0, str(v)) for v in point)


def point_rows(df: pd.DataFrame, point: tuple) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for field, value in zip(point_fields(df), point):
        mask &= df[field] == value
    return df[mask]


def point_label(df: pd.DataFrame, point: tuple, sep: str = ", ") -> str:
    return sep.join(f"{f}={v}" for f, v in zip(point_fields(df), point))


def variant_order_for(df: pd.DataFrame) -> List[str]:
    present = list(df["benchmark"].drop_duplicates())
    ordered = [v for v in BENCHMARK_ORDER if v in present]
    return ordered + [v for v in present if v not in ordered]


def variant_colors(variants: List[str]) -> dict:
    return {
        v: BENCHMARK_COLORS.get(v, FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])
        for i, v in enumerate(variants)
    }


def platform_order_for(df: pd.DataFrame) -> List[str]:
    # Sorted by the same PLATFORM_ORDER/COMPILER_ORDER precedence as plot_bars_per_cpu.py, so a
    # given machine sits in the same relative position across every cross-CPU chart in the project,
    # not just within this file.
    return sorted(df["platform"].unique(), key=platform_compiler_sort_key)


def plot_heatmap(df: pd.DataFrame, point: tuple, out_path: str, mode: str) -> None:
    """
    Rows = variant, columns = platform/compiler, color = ns_per_op relative to that column's own
    fastest variant (1.0 = winner on that platform). Normalizing per column (not globally) is
    what makes this a "which algorithm wins where" chart rather than a "which machine is fastest"
    chart -- raw clock-speed differences between platforms would otherwise dominate the color
    scale and hide the thing this chart is for.
    """
    sub = point_rows(df, point)
    variants = variant_order_for(sub)
    platforms = platform_order_for(sub)

    pivot = sub.pivot_table(index="benchmark", columns="platform", values="ns_per_op")
    pivot = pivot.reindex(index=variants, columns=platforms)
    ratio = pivot.div(pivot.min(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(1.4 * len(platforms) + 2.5, 0.55 * len(variants) + 1.5))
    im = ax.imshow(ratio.values, cmap="RdYlGn_r", vmin=1.0, vmax=2.0, aspect="auto")

    ax.set_xticks(range(len(platforms)))
    ax.set_xticklabels(platforms, rotation=30, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([label_for_mode(v, mode) for v in variants])

    for i in range(len(variants)):
        for j in range(len(platforms)):
            ns = pivot.values[i, j]
            r = ratio.values[i, j]
            if pd.isna(ns):
                continue
            label = f"{ns:.3f}\n({r:.2f}x)"
            color = "black" if r < 1.5 else "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ns/op, relative to fastest\non that platform", fontsize=9)

    ax.set_title(
        f"{point_label(df, point)} -- relative performance\n"
        "(1.0 = fastest variant on that platform)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_grouped_bars(df: pd.DataFrame, point: tuple, out_path: str, mode: str) -> None:
    """
    One group of bars per platform/compiler, one bar per variant within each group (same variant
    order/color as the heatmap), so bars for variants run on the same platform sit next to each
    other for direct comparison, and a given variant's color is consistent across every group.
    """
    sub = point_rows(df, point)
    variants = variant_order_for(sub)
    platforms = platform_order_for(sub)
    colors = variant_colors(variants)

    pivot = sub.pivot_table(index="platform", columns="benchmark", values="ns_per_op")
    pivot = pivot.reindex(index=platforms, columns=variants)

    n_variants = len(variants)
    group_width = 0.82
    bar_width = group_width / n_variants
    x = range(len(platforms))

    fig, ax = plt.subplots(figsize=(1.7 * len(platforms) * n_variants / 4 + 3, 6))
    for vi, variant in enumerate(variants):
        offsets = [xi - group_width / 2 + vi * bar_width + bar_width / 2 for xi in x]
        ax.bar(
            offsets, pivot[variant].values, width=bar_width,
            color=colors[variant], label=label_for_mode(variant, mode),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(platforms, rotation=30, ha="right")
    ax.set_ylabel("ns/op")
    ax.set_title(f"{point_label(df, point)} -- all variants, grouped by platform/compiler")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=min(n_variants, 5), fontsize=9,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )

    fig.subplots_adjust(bottom=0.42, top=0.92)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_platform_summary(df: pd.DataFrame, out_path: str) -> None:
    """
    One chart, separate from the per-variant comparisons above: for each platform/compiler, the
    best achievable ns/op (min across all variants), grouped into the case points. Answers
    "how do the machines/compilers compare to each other", deliberately kept apart from "which
    algorithm wins" so the two questions don't get conflated in one overloaded chart.
    """
    platforms = platform_order_for(df)
    points = case_points(df)
    group_width = 0.8
    bar_width = group_width / len(points)
    x = range(len(platforms))

    fig, ax = plt.subplots(figsize=(1.6 * len(platforms) + 3, 5.5))
    combo_colors = plt.get_cmap("tab10").colors
    for ci, point in enumerate(points):
        sub = point_rows(df, point)
        best = sub.groupby("platform")["ns_per_op"].min().reindex(platforms)
        offsets = [xi - group_width / 2 + ci * bar_width + bar_width / 2 for xi in x]
        ax.bar(
            offsets, best.values, width=bar_width, color=combo_colors[ci % len(combo_colors)],
            label=point_label(df, point),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(platforms, rotation=20, ha="right")
    ax.set_ylabel("best ns/op across all variants")
    ax.set_title("Best achievable performance per platform/compiler (any variant)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform/compiler comparison plots for the k-mer extraction suites"
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="CSV path of a k-mer suite. Can be given multiple times.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern(s) for the CSV files of a k-mer suite. Can be given multiple times.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--name", required=True,
        help="Start of the output file names, such as the suite name, or the name of a pair "
             "without the check",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=DEFAULT_KS,
        help="The values of k of the case points, for the suites with k (default: 29 32)",
    )
    add_mode_arg(parser)
    parser.add_argument(
        "--show-heatmaps",
        action="store_true",
        help="Also produce the per-(layout,k) relative-performance heatmaps (off by default -- "
             "the grouped bar charts and platform summary cover the same ground more usefully)",
    )
    args = parser.parse_args()

    files: List[str] = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))
    files = [f for f in files if f]

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = filter_mode(load_all(files, args.k), args.mode)
    # Not an error: e.g. a run with other values of k.
    if df.empty:
        warn(f"Nothing to plot: no rows with k in {args.k} of mode {args.mode} in {files}")
        return

    for point in case_points(df):
        tag = point_label(df, point, sep="_").replace("=", "")
        if args.show_heatmaps:
            out = os.path.join(out_dir, f"{args.name}_heatmap_{tag}_{args.mode}.png")
            plot_heatmap(df, point, out, args.mode)
        plot_grouped_bars(
            df, point, os.path.join(out_dir, f"{args.name}_bars_{tag}_{args.mode}.png"), args.mode
        )

    plot_platform_summary(
        df, os.path.join(out_dir, f"{args.name}_platform_summary_{args.mode}.png")
    )


if __name__ == "__main__":
    main()
