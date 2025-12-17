#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt


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

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract mask weight from "case" column, which looks like "popcount=17"
    df["weight"] = df["case"].str.split("=").str[1].astype(int)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for name, g in df.groupby("benchmark"):
        g = g.sort_values("weight")
        plt.plot(g["weight"], g["ns_per_op"], marker="o", label=name)

    plt.xlabel("Mask weight (popcount)")
    plt.ylabel("Time per operation [ns]")
    plt.title(args.title)

    plt.xlim(0, 64)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Implementation")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

