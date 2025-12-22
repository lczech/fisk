#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot k-mer extract benchmark results")
    parser.add_argument("csv", help="CSV file produced by bench_kmer_extract")
    parser.add_argument("--title", default="k-mer extract performance",
                        help="Plot title")
    parser.add_argument("--out", default=None,
                        help="Output image file (e.g. kmer_extract.png). If omitted, show interactively.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------
    df = pd.read_csv(args.csv)

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract k from "case" column, which looks like "k=17"
    df["k"] = df["case"].str.split("=").str[1].astype(int)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for name, g in df.groupby("benchmark"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["ns_per_op"], marker="o", label=name)

    plt.xlabel("k")
    plt.ylabel("Time per operation [ns]")
    plt.title(args.title)

    plt.xlim(1, 32)
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
