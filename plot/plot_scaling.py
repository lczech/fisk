#!/usr/bin/env python3
"""Plot the extractor scaling: the throughput of the tuned extractors over the length of the input,
one panel for the callback of one k-mer at a time, and one for the callback that also takes the
vectors of the SIMD algorithms.

With one CSV, the plot of one run: a line per series (k-mers and spaced k-mers, checked and
unchecked), and dashed lines at the sizes of the data caches, from the sys_info.txt next to the CSV.
The input is ASCII, so its length in bases is its size in bytes. With several CSVs, the plot of one
series across CPUs and compilers, see --series, with a line per CSV.

Usage examples:
  python plot/plot_scaling.py --file results/Intel_Core_7_240H/GCC_15/sum/extractor_scaling.csv \\
      --sink sum --out extractor_scaling.png

  python plot/plot_scaling.py --glob "results/*/*/sum/extractor_scaling.csv" \\
      --series kmer_checked --sink sum --out extractor_scaling_kmer_checked_per_cpu.png

If --out is omitted, shows interactively.
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '.')
from plot_common import (
    add_unit_scale_args,
    apply_unit_scale_suffix,
    apply_yscale,
    compute_axis_limits,
    convert_for_display,
    platform_compiler_sort_key,
    platform_from_csv_path,
    warn,
    ylabel_for_unit,
)

# The series, by the start of the names of their algorithms, as their label, color and line style:
# k-mers and spaced k-mers by color, checked and unchecked by line style.
SERIES = {
    "kmer_checked"          : ("k-mers, checked",          "tab:blue", "-"),
    "kmer_unchecked"        : ("k-mers, unchecked",        "tab:blue", "--"),
    "spaced_kmer_checked"   : ("spaced k-mers, checked",   "tab:red",  "-"),
    "spaced_kmer_unchecked" : ("spaced k-mers, unchecked", "tab:red",  "--"),
}
CALLBACKS = {
    "single"  : "One k-mer at a time",
    "vectors" : "With vectors",
}


def load(csv_path: str) -> pd.DataFrame:
    """The rows of one CSV, with the series, algorithm, callback, and length of each."""
    df = pd.read_csv(csv_path)
    names = df["benchmark"].str.extract(
        r"^for_each_(spaced_kmer|kmer)_ascii_(checked|unchecked)_(.*)$"
    )
    df["series"] = names[0] + "_" + names[1]
    df["algorithm"] = names[2]
    fields = df["case"].str.extract(r"callback=(\w+);length=(\d+)")
    df["callback"] = fields[0]
    df["length"] = fields[1].astype("int64")
    df["platform"] = platform_from_csv_path(csv_path)
    return df


def cache_sizes(csv_path: str) -> dict:
    """The data cache sizes in bytes, from the sys_info.txt next to the CSV, by level; empty when
    they are not there."""
    sizes = {}
    info = Path(csv_path).parent / "sys_info.txt"
    if not info.is_file():
        return sizes
    for line in info.read_text().splitlines():
        m = re.match(r"\s+(L1d|L2|L3)\s*:\s*(\d+) KiB", line)
        if m:
            sizes[m.group(1)] = int(m.group(2)) * 1024
    return sizes


def length_label(length: int) -> str:
    """64, 1 Ki, 16 Ki, ..., 4 Gi."""
    for shift, unit in ((30, "Gi"), (20, "Mi"), (10, "Ki")):
        if length >= (1 << shift):
            return f"{length >> shift} {unit}"
    return str(length)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the extractor scaling over the input length"
    )
    parser.add_argument(
        "--file", action="append", default=[], help="extractor_scaling.csv (repeatable)"
    )
    parser.add_argument(
        "--glob", action="append", default=[], help="Glob pattern of CSVs (repeatable)"
    )
    parser.add_argument(
        "--series", choices=list(SERIES), default=None,
        help="The series to plot across several CSVs (required for several)",
    )
    parser.add_argument(
        "--min-length", type=int, default=1024,
        help="Shortest input length to show (default: 1024); shorter ones show mostly the fixed "
             "cost per call",
    )
    parser.add_argument(
        "--out", default=None, help="Output image file; if omitted, show interactively"
    )
    add_unit_scale_args(parser)
    args = parser.parse_args()

    files = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        warn("No CSV files given, nothing to plot")
        return
    across = len(files) > 1
    if across and args.series is None:
        parser.error("--series is required for several CSVs")

    df = pd.concat([load(f) for f in files], ignore_index=True)
    if args.series is not None:
        df = df[df["series"] == args.series]
    shortest = df["length"].min()
    df = df[df["length"] >= args.min_length]
    if df.empty:
        warn(f"Nothing to plot in {files}")
        return
    df["display_value"] = convert_for_display(df["ns_per_op"], args.unit)

    fig, axes = plt.subplots(1, len(CALLBACKS), figsize=(14, 5.5), sharey=True)
    platforms = sorted(df["platform"].unique(), key=platform_compiler_sort_key)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (callback, title) in zip(axes, CALLBACKS.items()):
        part = df[df["callback"] == callback]
        if across:
            lines = [
                (platform, part[part["platform"] == platform], colors[i % len(colors)], "-")
                for i, platform in enumerate(platforms)
            ]
        else:
            lines = [
                (label, part[part["series"] == key], color, style)
                for key, (label, color, style) in SERIES.items()
            ]
        for label, rows, color, style in lines:
            if rows.empty:
                continue
            rows = rows.sort_values("length")
            ax.plot(
                rows["length"], rows["display_value"], marker=".", color=color, linestyle=style,
                linewidth=2, label=f"{label} ({rows['algorithm'].iloc[0]})",
            )

        # The cache sizes of the CPU of a single run.
        if not across:
            for level, size in cache_sizes(files[0]).items():
                ax.axvline(size, color="gray", linestyle=":", linewidth=1.5)
                ax.text(size, 1.0, f" {level}", transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=11, color="gray")

        ax.set_xscale("log", base=2)
        ticks = [
            1 << s for s in range(6, 33, 2)
            if df["length"].min() <= (1 << s) <= df["length"].max()
        ][::2]
        ax.set_xticks(ticks, [length_label(t) for t in ticks])
        ax.set_xlabel("Input length [bases]")
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")

    apply_yscale(axes[0], args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale)
    axes[0].set_ylim(args.y_min if args.y_min is not None else y_min,
                     args.y_max if args.y_max is not None else y_max)
    axes[0].set_ylabel(ylabel_for_unit(args.unit, "extractor_scaling"))
    fig.suptitle(SERIES[args.series][0] if across else platforms[0])
    if shortest < args.min_length:
        fig.text(
            0.5, 0.01, f"From {length_label(args.min_length)} bases on; for shorter inputs, the "
            "fixed cost per call lowers the throughput.", ha="center", va="bottom", fontsize=11,
            color="gray",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale, args.sink)
    if out:
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
