#!/usr/bin/env python3
"""
Cross-platform benchmark summary plot.

Reads multiple benchmark CSV files (same schema: suite,case,benchmark,ns_per_op),
computes per-(benchmark, platform) mean over cases, and plots a grouped bar chart:

- x-axis: benchmark (implementation name)
- for each benchmark: bars for each platform (CPU), colored by platform
- y-axis: mean ns/op across cases

Platform label is taken from the last path component of the directory containing the CSV.
Example: results/Epyc/pext.csv -> platform label "Epyc"

Usage examples:
  python plot_cross_platform.py --file results/Epyc/pext.csv --file results/Ryzen/pext.csv --suite PEXT --out pext_platforms.png
  python plot_cross_platform.py --glob "artifacts/*/pext.csv" --suite PEXT

If --out is omitted, shows interactively.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def platform_from_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    # "last part of the directory path name"
    # For ".../<CPU>/<file>.csv" this returns "<CPU>"
    return p.parent.name


def load_one(csv_path: str, platform: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if platform is None:
        platform = platform_from_csv_path(csv_path)
    df["platform"] = platform
    df["source"] = str(csv_path)

    PLOT_BENCHMARKS = {
        # "char_to_nt_ifs_re",
        # "char_to_nt_switch_re",
        # "char_to_nt_table_re",
        # "char_to_nt_ascii_re",
        "char_to_nt_ifs",
        "char_to_nt_switch",
        "char_to_nt_ascii",
        "char_to_nt_table",
    }
    df = df[df["benchmark"].isin(PLOT_BENCHMARKS)]

    return df

def color_for_platform(platform: str) -> str:
    PLATFORM_COLORS = [
        ("epyc",   "#08aa72"),  # green
        ("ryzen",  "#fb6516"),  # orange
        ("xeon",   "#0874d1"),  # blue
        ("m1",     "#979797"),  # gray
        ("m2",     "#979797"),  # gray
    ]

    p = platform.lower()
    for key, color in PLATFORM_COLORS:
        if key in p:
            return color
    return "grey"  # fallback

def main() -> None:
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
        "--title",
        default=None,
        help="Plot title override.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figure (default: 200).",
    )
    args = ap.parse_args()

    files: list[str] = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))

    files = [f for f in files if f]  # drop empty strings
    if not files:
        raise SystemExit("No input files. Provide --file ... or --glob ...")

    # Load all
    dfs = [load_one(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Basic validation
    required_cols = {"suite", "case", "benchmark", "ns_per_op", "platform"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in input CSV(s): {sorted(missing)}")

    # Optional suite filter
    suite = args.suite
    if suite is not None:
        df = df[df["suite"] == suite]
        if df.empty:
            available = sorted(set(pd.concat(dfs)["suite"].astype(str)))
            raise SystemExit(f"No rows for suite='{suite}'. Available suites: {available}")

    # Aggregate: mean over cases for each (benchmark, platform)
    agg = (
        df.groupby(["benchmark", "platform"], as_index=False)["ns_per_op"]
          .mean()
          .rename(columns={"ns_per_op": "mean_ns_per_op"})
    )

    # Pivot for plotting: rows=benchmark, columns=platform
    pivot = agg.pivot(index="benchmark", columns="platform", values="mean_ns_per_op")

    # Stable ordering
    PLATFORM_ORDER = [
        "Epyc",
        "Ryzen",
        "Xeon",
        "M1",
        "M2",
    ]
    platforms = list(pivot.columns)
    # platforms = [p for p in PLATFORM_ORDER if p in pivot.columns]

    BENCHMARK_ORDER = [
        "char_to_nt_ifs",
        "char_to_nt_switch",
        "char_to_nt_ascii",
        "char_to_nt_table",
    ]
    # benchmarks = list(pivot.index)
    # benchmarks = [b for b in BENCHMARK_ORDER if b in pivot.index]

    pivot = pivot.reindex(BENCHMARK_ORDER)
    benchmarks = pivot.index.tolist()

    # Plot
    import numpy as np

    x = np.arange(len(benchmarks))
    group_width = 0.8
    bar_w = group_width / max(1, len(platforms))

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(benchmarks) + 4), 6))

    # for j, plat in enumerate(platforms):
    #     vals = pivot[plat].values
    #     # If some platform lacks a benchmark, vals may contain NaN. Bar will skip those.
    #     ax.bar(x + (j - (len(platforms) - 1) / 2.0) * bar_w, vals, width=bar_w, label=str(plat))

    for j, plat in enumerate(platforms):
        vals = pivot[plat].values
        ax.bar(
            x + (j - (len(platforms) - 1) / 2.0) * bar_w,
            vals,
            width=bar_w,
            label=str(plat),
            color=color_for_platform(plat),
        )


    title = args.title
    if title is None:
        title = f"{suite} cross-platform summary" if suite else "Cross-platform summary"

    # ax.set_title(title)
    # ax.set_xlabel("Implementation")
    # ax.set_ylabel("Time per operation [ns] across cases")
    ax.set_ylabel("Time per operation [ns]")

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # ax.legend(title="Platform (CPU)")
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace("_", " ") for label in labels]
    ax.legend(
        handles,
        labels,
        title="Platform (CPU)",
    )

    fig.tight_layout()

    if args.out:
        out = args.out
        os.makedirs(str(Path(out).parent), exist_ok=True) if str(Path(out).parent) not in ("", ".") else None
        fig.savefig(out, dpi=args.dpi)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
