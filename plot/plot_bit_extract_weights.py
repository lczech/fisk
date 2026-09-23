#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
from pathlib import Path
from typing import Dict, List, Optional

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


def _label_for_benchmark(name: str, rename_map: Dict[str, str]) -> str:
    return rename_map.get(name, name)


def main():
    parser = argparse.ArgumentParser(description="Plot PEXT benchmark results")
    parser.add_argument("csv", help="CSV file produced by bench_pext")
    parser.add_argument("--title", default="PEXT performance vs mask weight",
                        help="Plot title")
    parser.add_argument("--out", default=None,
                        help="Output image file (e.g. pext.png). If omitted, show interactively.")
    add_unit_scale_args(parser)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    df = pd.read_csv(args.csv)
    cpu = platform_from_csv_path(args.csv)
    suite = df["suite"].iloc[0] if not df.empty else None

    # Subset to the benchmarks we want to plot
    df = df[df["benchmark"].isin(BENCHMARKS_KEEP_EXTENDED)]

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract mask weight from "case" column, which looks like "popcount=17"
    df["weight"] = df["case"].str.split("=").str[1].astype(int)

    # Convert to the requested display unit once, up front -- no aggregation happens in this
    # script (each row is already one (benchmark, weight) point), so there's no ordering concern
    # between conversion and aggregation here.
    df["display_value"] = convert_for_display(df["ns_per_op"], args.unit)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    # for name, g in df.groupby("benchmark"):
    #     g = g.sort_values("weight")
    #     plt.plot(g["weight"], g["ns_per_op"], marker="", label=name, linewidth=2)

    for name in BENCHMARK_ORDER:
        g = df[df["benchmark"] == name]
        if g.empty:
            continue
        g = g.sort_values("weight")
        color = BENCHMARK_COLORS.get(name, "black")
        ax.plot(
            g["weight"],
            g["display_value"],
            marker=".",
            label=_label_for_benchmark(name, BENCHMARK_RENAMES),
            color=color,
            linewidth=2,
        )

    # If pext is missing, insert an invisible placeholder at the front
    # so the 2-column legend keeps the same visual grouping.
    handles, labels = ax.get_legend_handles_labels()
    if "pext" not in set(df["benchmark"]):
        handles = [Line2D([], [], linestyle="none", marker=None, alpha=0)] + handles
        labels = [""] + labels

    ax.set_xlabel("Mask weight (popcount)")
    ax.set_ylabel(ylabel_for_unit(args.unit, suite))
    ax.set_title(cpu.replace("_", " "))
    # ax.set_title(args.title)

    ax.set_xlim(0, 64)
    apply_yscale(ax, args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale)
    if args.y_min is not None:
        y_min = args.y_min
    if args.y_max is not None:
        y_max = args.y_max
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    # ax.legend(title="Implementation", ncol=2)
    # ax.legend(ncol=2)
    ax.legend(handles, labels, ncol=2, loc="upper right")

    fig.tight_layout()

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    if out:
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
