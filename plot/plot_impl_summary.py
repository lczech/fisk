#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def make_impl_summary_plot(df, suite, title, outpath):
    # Filter for this suite (if given)
    if suite is not None:
        df = df[df["suite"] == suite]

    if df.empty:
        raise ValueError(f"No data found for suite: {suite}")

    # Aggregate per implementation (benchmark) across all cases
    grouped = (
        df.groupby("benchmark")["ns_per_op"]
          .agg(["mean", "min", "max"])
          .reset_index()
    )

    impls = grouped["benchmark"].tolist()
    means = grouped["mean"].values
    mins  = grouped["min"].values
    maxs  = grouped["max"].values

    # Asymmetric error bars: mean - min, max - mean
    lower_err = means - mins
    upper_err = maxs - means

    import numpy as np
    x = np.arange(len(impls))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars at the mean
    bars = ax.bar(x, means, width=width, label="mean ns/op")

    # Add whiskers for min/max
    ax.errorbar(
        x,
        means,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=4,
        label="min/max"
    )

    ax.set_title(title)
    ax.set_xlabel("Implementation")
    ax.set_ylabel("ns/op (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Saved: {outpath}")
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
        "--out",
        default=None,
        help="Output image path (if omitted, show interactively)",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title override (default: suite name or generic)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    suite = args.suite
    title = args.title or (suite if suite else "Implementation summary")

    make_impl_summary_plot(df, suite, title, args.out)


if __name__ == "__main__":
    main()
