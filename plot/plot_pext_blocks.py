#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def platform_from_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    # "last part of the directory path name"
    # For ".../<CPU>/<file>.csv" this returns "<CPU>"
    return p.parent.name


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

    PLOT_BENCHMARKS = {
        "pext_hw_bmi2",
        "pext_sw_bitloop",
        # "pext_sw_split32",
        "pext_sw_table8",
        "pext_sw_adaptive",
        "pext_sw_block_table",
        "pext_sw_block_table_unrolled2",
        "pext_sw_block_table_unrolled4",
        "pext_sw_block_table_unrolled8",
        # "pext_sw_instlatx",
        # "pext_sw_zp7",
    }
    df = df[df["benchmark"].isin(PLOT_BENCHMARKS)]

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract mask weight from "case" column, which looks like "popcount=17"
    df["weight"] = df["case"].str.split("=").str[1].astype(int)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    # Stable colors for each implementation / benchmark
    BENCHMARK_COLORS = {
        "pext_hw_bmi2":        "#5F5F5F",
        "pext_sw_bitloop":     "#47a1e2",
        "pext_sw_split32":     "#ff7f0e",
        "pext_sw_table8":      "#884ed3",
        "pext_sw_adaptive":    "#D35820",
        "pext_sw_block_table":           "#a1d99b",
        "pext_sw_block_table_unrolled2": "#74c476",
        "pext_sw_block_table_unrolled4": "#31a354",
        "pext_sw_block_table_unrolled8": "#006d2c",
        "pext_sw_instlatx":    "#000000",
        "pext_sw_zp7":         "#000000",
    }
    LINE_ORDER = [
        "pext_hw_bmi2",
        "pext_sw_bitloop",
        "pext_sw_split32",
        "pext_sw_table8",
        "pext_sw_adaptive",
        "pext_sw_block_table",
        "pext_sw_block_table_unrolled2",
        "pext_sw_block_table_unrolled4",
        "pext_sw_block_table_unrolled8",
        "pext_sw_instlatx",
        "pext_sw_zp7",
    ]

    plt.figure(figsize=(8, 5))

    # for name, g in df.groupby("benchmark"):
    #     g = g.sort_values("weight")
    #     plt.plot(g["weight"], g["ns_per_op"], marker="", label=name, linewidth=2)

    for name in LINE_ORDER:
        g = df[df["benchmark"] == name]
        if g.empty:
            continue
        g = g.sort_values("weight")
        color = BENCHMARK_COLORS.get(name, "black")
        plt.plot(
            g["weight"],
            g["ns_per_op"],
            marker=".",
            label=name,
            color=color,
            linewidth=2,
        )

    plt.xlabel("Number of blocks (runs of 1s)")
    plt.ylabel("Time per operation [ns]")
    plt.title(cpu.replace("_", " "))
    # plt.title(args.title)

    plt.xlim(0, 32)
    plt.ylim(0, 16)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Implementation", ncol=2)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
