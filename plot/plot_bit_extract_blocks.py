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
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    df = pd.read_csv(args.csv)
    cpu = platform_from_csv_path(args.csv)

    # Subset to the benchmarks we want to plot
    df = df[df["benchmark"].isin(BENCHMARKS_KEEP_EXTENDED)]

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract mask weight from "case" column, which looks like "popcount=17"
    df["weight"] = df["case"].str.split("=").str[1].astype(int)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # for name, g in df.groupby("benchmark"):
    #     g = g.sort_values("weight")
    #     plt.plot(g["weight"], g["ns_per_op"], marker="", label=name, linewidth=2)

    for name in BENCHMARK_ORDER:
        g = df[df["benchmark"] == name]
        if g.empty:
            continue
        g = g.sort_values("weight")
        color = BENCHMARK_COLORS.get(name, "black")
        plt.plot(
            g["weight"],
            g["ns_per_op"],
            marker=".",
            label=_label_for_benchmark(name, BENCHMARK_RENAMES),
            color=color,
            linewidth=2,
        )

    # If pext is missing, insert an invisible placeholder at the front
    # so the 2-column legend keeps the same visual grouping.
    handles, labels = plt.gca().get_legend_handles_labels()
    if "pext" not in set(df["benchmark"]):
        handles = [Line2D([], [], linestyle="none", marker=None, alpha=0)] + handles
        labels = [""] + labels

    plt.xlabel("Number of blocks (runs of 1s)")
    plt.ylabel("Time per operation [ns]")
    plt.title(cpu.replace("_", " "))
    # plt.title(args.title)

    plt.xlim(0, 32)
    # plt.ylim(0, 16)
    plt.ylim(0, 40)
    # plt.ylim(0, 25)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    # plt.legend(title="Implementation")
    # plt.legend(ncol=2)
    # plt.legend()
    plt.legend(handles, labels, ncol=2, loc="upper right")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
