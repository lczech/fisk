#!/usr/bin/env python3

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
    BENCHMARK_RENAMES,
    PACKED_KMER_VARIANT_COLORS,
    PACKED_KMER_VARIANT_ORDER,
    parse_case_fields,
    platform_compiler_sort_key,
    platform_from_csv_path,
)

# Order and colors come from plot_common.py, shared with plot_kmer_extract_packed.py, so a given
# variant's color is identical across every plot in this whole family, not just within this file.
VARIANT_ORDER = PACKED_KMER_VARIANT_ORDER
VARIANT_COLORS = PACKED_KMER_VARIANT_COLORS

# Direct narrow/wide rows are useful while tuning, but the dispatcher rows are the default user-facing
# comparison. Pass --show-direct to include the helper rows as well.
DIRECT_SIMD_VARIANTS = {
    v for v in VARIANT_ORDER if v.startswith("simd_") and ("_narrow_" in v or "_wide_" in v)
}
ROLLING_VARIANTS = {"rolling", "narrow_rolling", "wide_rolling"}

# (layout, k) combos to produce one heatmap + one bar chart for each.
COMBOS = [("msb", 29), ("msb", 32), ("lsb", 29), ("lsb", 32)]


def load_all(csv_paths: List[str]) -> pd.DataFrame:
    """
    Load each given kmer_extract_packed.csv, tag each row with its platform (the parent directory
    name, e.g. "AMD EPYC 9684X, Clang 17"), and concatenate into one long-format DataFrame. Takes
    an explicit file list (built by the caller, e.g. from plot_all_cpus.sh's curated CPUS array via
    --file, same as plot_bars_per_cpu.py) rather than globbing a results directory itself, so the
    set of platforms shown here always matches the rest of the cross-CPU plots -- not a second,
    independently-curated selection.
    """
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = parse_case_fields(df)
        df["k"] = df["k"].astype(int)
        df["platform"] = platform_from_csv_path(csv_path)
        frames.append(df)
    if not frames:
        raise ValueError("No input files. Provide --file ... or --glob ...")
    return pd.concat(frames, ignore_index=True)


def variant_order_for(df: pd.DataFrame) -> List[str]:
    present = set(df["benchmark"])
    return [v for v in VARIANT_ORDER if v in present]


def filter_variants(
    df: pd.DataFrame, *, show_rolling: bool, show_direct: bool
) -> pd.DataFrame:
    """Apply the plot switches once so every comparison chart uses the same exclusions."""
    excluded = set() if show_rolling else set(ROLLING_VARIANTS)
    if not show_direct:
        excluded |= DIRECT_SIMD_VARIANTS
    return df[~df["benchmark"].isin(excluded)]


def platform_order_for(df: pd.DataFrame) -> List[str]:
    # Sorted by the same PLATFORM_ORDER/COMPILER_ORDER precedence as plot_bars_per_cpu.py, so a
    # given machine sits in the same relative position across every cross-CPU chart in the project,
    # not just within this file.
    return sorted(df["platform"].unique(), key=platform_compiler_sort_key)


def plot_heatmap(df: pd.DataFrame, layout: str, k: int, out_path: str) -> None:
    """
    Rows = variant, columns = platform/compiler, color = ns_per_op relative to that column's own
    fastest variant (1.0 = winner on that platform). Normalizing per column (not globally) is
    what makes this a "which algorithm wins where" chart rather than a "which machine is fastest"
    chart -- raw clock-speed differences between platforms would otherwise dominate the color
    scale and hide the thing this chart is for.
    """
    sub = df[(df["layout"] == layout) & (df["k"] == k)]
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
    ax.set_yticklabels([BENCHMARK_RENAMES.get(v, v) for v in variants])

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
        f"layout={layout}, k={k} -- relative performance\n(1.0 = fastest variant on that platform)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_grouped_bars(df: pd.DataFrame, layout: str, k: int, out_path: str) -> None:
    """
    One group of bars per platform/compiler, one bar per variant within each group (same variant
    order/color as the heatmap), so bars for variants run on the same platform sit next to each
    other for direct comparison, and a given variant's color is consistent across every group.
    """
    sub = df[(df["layout"] == layout) & (df["k"] == k)]
    variants = variant_order_for(sub)
    platforms = platform_order_for(sub)

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
            color=VARIANT_COLORS[variant], label=BENCHMARK_RENAMES.get(variant, variant),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(platforms, rotation=30, ha="right")
    ax.set_ylabel("ns/op")
    ax.set_title(f"layout={layout}, k={k} -- all variants, grouped by platform/compiler")
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
    best achievable ns/op (min across all variants), grouped into the 4 layout/k combos. Answers
    "how do the machines/compilers compare to each other", deliberately kept apart from "which
    algorithm wins" so the two questions don't get conflated in one overloaded chart.
    """
    platforms = platform_order_for(df)
    group_width = 0.8
    bar_width = group_width / len(COMBOS)
    x = range(len(platforms))

    fig, ax = plt.subplots(figsize=(1.6 * len(platforms) + 3, 5.5))
    combo_colors = plt.get_cmap("tab10").colors
    for ci, (layout, k) in enumerate(COMBOS):
        sub = df[(df["layout"] == layout) & (df["k"] == k)]
        best = sub.groupby("platform")["ns_per_op"].min().reindex(platforms)
        offsets = [xi - group_width / 2 + ci * bar_width + bar_width / 2 for xi in x]
        ax.bar(offsets, best.values, width=bar_width, color=combo_colors[ci], label=f"{layout}, k={k}")

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
        description="Cross-platform/compiler comparison plots for kmer_extract_packed benchmarks"
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="kmer_extract_packed.csv path. Can be given multiple times.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern(s) for kmer_extract_packed.csv files. Can be given multiple times.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--show-rolling",
        action="store_true",
        help="Include rolling reference implementations (hidden by default)",
    )
    parser.add_argument(
        "--show-direct",
        action="store_true",
        help="Include direct narrow/wide SIMD helper rows (hidden by default)",
    )
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

    df = filter_variants(
        load_all(files), show_rolling=args.show_rolling, show_direct=args.show_direct
    )

    for layout, k in COMBOS:
        if args.show_heatmaps:
            plot_heatmap(
                df, layout, k,
                os.path.join(out_dir, f"kmer_extract_packed_compare_heatmap_{layout}_k{k}.png"),
            )
        plot_grouped_bars(
            df, layout, k,
            os.path.join(out_dir, f"kmer_extract_packed_compare_bars_{layout}_k{k}.png"),
        )

    plot_platform_summary(
        df, os.path.join(out_dir, "kmer_extract_packed_compare_platform_summary.png")
    )


if __name__ == "__main__":
    main()
