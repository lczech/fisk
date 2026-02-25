#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def make_grouped_bar_plot_impl_first(df, suite, title, outpath):
    # Filter for this suite (if given)
    if suite is not None:
        df = df[df["suite"] == suite]

    if df.empty:
        raise ValueError(f"No data found for suite: {suite}")

    # Pivot so index = implementation (benchmark), columns = case
    # Each row will be one implementation; columns are the cases
    pivot = df.pivot_table(
        index="benchmark",
        columns="case",
        values="ns_per_op",
        aggfunc="mean",  # in case there are duplicates
    )

    impls = list(pivot.index)    # implementations on x-axis
    cases = list(pivot.columns)  # one bar per case within each impl group

    num_impls = len(impls)
    num_cases = len(cases)

    import numpy as np

    # X positions: one group per implementation
    x = np.arange(num_impls)

    # Width of each bar inside a group
    group_width = 0.8
    bar_width = group_width / max(1, num_cases)

    # Center the bars within each group
    # For case j, offset is (j - (num_cases - 1)/2) * bar_width
    offsets = [
        (j - (num_cases - 1) / 2.0) * bar_width
        for j in range(num_cases)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    # One color series per case
    for j, case in enumerate(cases):
        vals = pivot[case].values
        ax.bar(x + offsets[j], vals, width=bar_width, label=str(case))

    ax.set_title(title)
    ax.set_xlabel("Implementation")
    ax.set_ylabel("ns/op (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=45, ha="right")

    ax.legend(title="Case")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Wrote {outpath}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Grouped bar plot: all cases per implementation."
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
    title = args.title or (suite if suite else "Grouped Benchmarks (impl-first)")

    make_grouped_bar_plot_impl_first(df, suite, title, args.out)


if __name__ == "__main__":
    main()
