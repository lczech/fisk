#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os, sys

# Such a cheat to import stuff in python...
# See https://stackoverflow.com/a/22956038
sys.path.insert(0, '.')
from plot_common import *


def _ordered_benchmarks(present: List[str], order: Optional[List[str]]) -> List[str]:
    """
    Return benchmarks in the desired plotting order:
    - first, those in 'order' that are present
    - then, any remaining benchmarks in their current order
    """
    if not order:
        return present
    order_present = [b for b in order if b in present]
    remaining = [b for b in present if b not in set(order_present)]
    return order_present + remaining


def _color_for_benchmark(name: str, color_map: Dict[str, str]) -> Optional[str]:
    """
    Return a stable color if provided; otherwise None (Matplotlib will pick from cycle).
    """
    return color_map.get(name)


def _linestyle_for_benchmark(name: str, linestyle_map: Dict[str, str]) -> Optional[str]:
    """
    Return a stable linestyle if provided; otherwise None (Matplotlib default "-").
    """
    return linestyle_map.get(name)


def _label_for_benchmark(name: str, rename_map: Dict[str, str]) -> str:
    return rename_map.get(name, name)


def main():
    parser = argparse.ArgumentParser(description="Plot k-mer extract benchmark results")
    parser.add_argument("csv", help="CSV file produced by bench_kmer_extract")
    parser.add_argument("--title", default="k-mer extract performance",
                        help="Plot title")
    parser.add_argument("--out", default=None,
                        help="Output image file (e.g. kmer_extract.png). If omitted, show interactively.")
    add_unit_scale_args(parser)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    df = pd.read_csv(args.csv)
    cpu = platform_from_csv_path(args.csv)
    suite = df["suite"].iloc[0] if not df.empty else None

    # Expect columns:
    #   suite, case, benchmark, ns_per_op
    #
    # Extract k from "case" column, which looks like "k=17"
    df["k"] = df["case"].str.split("=").str[1].astype(int)

    # Optional: filter benchmarks
    if BENCHMARKS_KEEP is not None:
        df = df[df["benchmark"].isin(BENCHMARKS_KEEP)]

    if df.empty:
        raise ValueError(
            f"No rows left after filtering with BENCHMARKS_KEEP={BENCHMARKS_KEEP!r} "
            f"in CSV file {args.csv!r}"
        )

    # Convert to the requested display unit once, up front -- no aggregation happens in this
    # script (each row is already one (benchmark, k) point), so there's no ordering concern
    # between conversion and aggregation here.
    df["display_value"] = convert_for_display(df["ns_per_op"], args.unit)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine plotting order for lines/legend
    present = df["benchmark"].drop_duplicates().tolist()
    plot_order = _ordered_benchmarks(present, BENCHMARK_ORDER)

    # for name in plot_order:
    #     g = df[df["benchmark"] == name].sort_values("k")
    #     color = _color_for_benchmark(name, BENCHMARK_COLORS)
    #     if color is None:
    #         plt.plot(g["k"], g["ns_per_op"], marker="o", label=name)
    #     else:
    #         plt.plot(g["k"], g["ns_per_op"], marker="o", label=name, color=color)

    for name in plot_order:
        g = df[df["benchmark"] == name].sort_values("k")

        color = _color_for_benchmark(name, BENCHMARK_COLORS)
        linestyle = _linestyle_for_benchmark(name, BENCHMARK_LINESTYLES)

        plot_kwargs = {
            "marker": ".",
            # "markersize": 4,
            "linewidth": 2,
            "label": _label_for_benchmark(name, BENCHMARK_RENAMES),
        }

        if color is not None:
            plot_kwargs["color"] = color
        if linestyle is not None:
            plot_kwargs["linestyle"] = linestyle

        ax.plot(g["k"], g["display_value"], **plot_kwargs)

    ax.set_xlabel("k-mer size (k)")
    ax.set_ylabel(ylabel_for_unit(args.unit, suite))
    ax.set_title(cpu.replace("_", " "))
    # ax.set_title(args.title)

    ax.set_xlim(1, 32)
    apply_yscale(ax, args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale)
    if args.y_min is not None:
        y_min = args.y_min
    if args.y_max is not None:
        y_max = args.y_max
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    # ax.legend(title="Implementation")
    ax.legend(loc="upper left", handlelength=2.75)

    fig.tight_layout()

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    if out:
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
