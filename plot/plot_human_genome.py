#!/usr/bin/env python3
"""Plot the example on the human genome across CPUs and compilers: the throughput of for_each_kmer()
and for_each_spaced_kmer(), one panel for the k-mers and one for the spaced k-mers, with the CPUs on
the x-axis and a bar per compiler, colored by CPU and hatched by compiler, as in
plot_bars_per_cpu.py. Each bar is labeled with the algorithm of the estimate. The time to read the
genome is not shown.

Reads the human_genome.csv of each CPU and compiler, which scripts/run_benchmarks.sh writes next to
the sink folders, see example/run_human_genome.sh. With --csv, also writes all of them as one table,
with the CPU and compiler first, and the time per k-mer and the k-mers per second added. Warns if the
k-mer counts or checksums of an extraction differ between the files, as all runs read the same
genome.

Usage example:
  python plot/plot_human_genome.py \\
      --file results/AMD_EPYC_9655/Clang_21/human_genome.csv \\
      --file results/AMD_EPYC_9655/GCC_15/human_genome.csv \\
      --csv human_genome.csv --out human_genome_per_cpu.png

If neither --out nor --csv is given, shows interactively.
"""

import argparse
import glob
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '.')
from plot_common import (
    COMPILER_ORDER,
    PLATFORM_ORDER,
    add_unit_scale_args,
    apply_unit_scale_suffix,
    apply_yscale,
    benchmark_label,
    color_for_platform,
    compute_axis_limits,
    convert_for_display,
    hatch_for_compiler,
    infer_compiler,
    infer_platform,
    platform_from_csv_path,
    warn,
    ylabel_for_unit,
)

mpl.rcParams["hatch.linewidth"] = 0.7

# The extractions of the example, as the panels, by their name in the CSV.
EXTRACTIONS = {
    "kmers"        : "k-mers",
    "spaced_kmers" : "Spaced k-mers",
}


def load(csv_path: str) -> pd.DataFrame:
    """The rows of one CSV, with its CPU and compiler first, and the time per k-mer and k-mers per
    second added."""
    df = pd.read_csv(csv_path)
    cpu, _, compiler = platform_from_csv_path(csv_path).rpartition(", ")
    df.insert(0, "compiler", compiler)
    df.insert(0, "cpu", cpu)
    df["ns_per_kmer"] = df["extract_seconds"] * 1e9 / df["kmers"]
    df["kmers_per_second"] = df["kmers"] / df["extract_seconds"]
    return df


def check_consistent(df: pd.DataFrame) -> None:
    """Warn if the k-mer counts or checksums of an extraction differ between the runs."""
    for extraction, rows in df.groupby("extraction"):
        for column, name in (("kmers", "k-mer counts"), ("checksum", "checksums")):
            if rows[column].nunique() > 1:
                runs = ", ".join(
                    f"{r.cpu} {r.compiler}: {getattr(r, column)}" for r in rows.itertuples()
                )
                warn(f"The {name} of {extraction} differ between the runs: {runs}")


def algorithm_label(algorithm: str) -> str:
    """The short label of an algorithm of the extractions, such as "PEXT" for
    for_each_spaced_kmer_ascii_checked_chunked_scalar_pext, or "Butterfly AVX512" for the SIMD
    butterfly: the input and check are those of the example for all of them, all algorithms from
    ASCII are chunked, and the extraction of spaced k-mers names the technique."""
    name = algorithm.removeprefix("for_each_").removeprefix("spaced_").removeprefix("kmer_")
    name = name.removeprefix("ascii_").removeprefix("checked_").removeprefix("unchecked_")
    return benchmark_label(name).removeprefix("Chunked ").removeprefix("scalar, ").removeprefix(
        "SIMD, ")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the example on the human genome across CPUs and compilers"
    )
    parser.add_argument(
        "--file", action="append", default=[], help="human_genome.csv (repeatable)"
    )
    parser.add_argument(
        "--glob", action="append", default=[], help="Glob pattern of CSVs (repeatable)"
    )
    parser.add_argument(
        "--csv", default=None, help="Output CSV file of all runs together (optional)"
    )
    parser.add_argument(
        "--out", default=None, help="Output image file; if omitted, shows interactively, unless "
        "--csv is given"
    )
    add_unit_scale_args(parser)
    args = parser.parse_args()

    files = list(args.file)
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        warn("No CSV files given, nothing to plot")
        return

    df = pd.concat([load(f) for f in files], ignore_index=True)
    df["platform"] = df["cpu"].map(infer_platform)
    df["compiler_family"] = df["compiler"].map(infer_compiler)
    check_consistent(df)
    if args.csv:
        df.drop(columns=["platform", "compiler_family"]).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
        if args.out is None:
            return
    df["display_value"] = convert_for_display(df["ns_per_kmer"], args.unit)

    # The CPUs and compilers in the order of all other plots across CPUs, see plot_common.py.
    def order(names, family, known):
        return sorted(names, key=lambda n: (
            known.index(family(n)) if family(n) in known else len(known), n
        ))
    cpus = order(df["cpu"].unique(), infer_platform, PLATFORM_ORDER)
    compilers = order(df["compiler"].unique(), infer_compiler, COMPILER_ORDER)

    fig, axes = plt.subplots(
        1, len(EXTRACTIONS), figsize=(max(8, 2.2 * len(cpus) * len(EXTRACTIONS)), 6), sharey=True
    )
    width = 0.8 / len(compilers)
    labels = []  # (axes, label, x, value) of each bar, to move the labels that do not fit
    for ax, (extraction, title) in zip(axes, EXTRACTIONS.items()):
        part = df[df["extraction"] == extraction]
        for i, cpu in enumerate(cpus):
            for j, compiler in enumerate(compilers):
                row = part[(part["cpu"] == cpu) & (part["compiler"] == compiler)]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = i - 0.4 + width * (j + 0.5)
                ax.bar(
                    x, row["display_value"], width=width,
                    color=color_for_platform(row["platform"]),
                    hatch=hatch_for_compiler(row["compiler_family"]),
                    edgecolor="white", linewidth=0, zorder=3,
                )
                # The algorithm of the estimate, upwards from the bottom of the bar.
                label = ax.text(
                    x, 0.03, algorithm_label(row["algorithm"]), transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=10, color="white", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.35,
                              linewidth=0),
                )
                labels.append((ax, label, x, row["display_value"]))
        ax.set_xticks(range(len(cpus)), [infer_platform(c) if infer_platform(c) != "Other" else c
                                          for c in cpus])
        ax.set_xlim(-0.6, len(cpus) - 0.4)
        ax.set_title(title)
        ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.5, zorder=0)

    # The compilers by their hatch, in gray, as the colors are those of the CPUs.
    handles = [
        mpl.patches.Patch(facecolor="#979797", hatch=hatch_for_compiler(infer_compiler(c)),
                          edgecolor="white", linewidth=0, label=c)
        for c in compilers
    ]
    # Below the title, so that it covers no bar.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(handles), frameon=False)

    # On a log scale, room below the shortest bar for its label.
    apply_yscale(axes[0], args.scale)
    y_min, y_max = compute_axis_limits(df["display_value"], args.scale, headroom=1.15)
    if args.scale == "log":
        y_min /= 3
    axes[0].set_ylim(args.y_min if args.y_min is not None else y_min,
                     args.y_max if args.y_max is not None else y_max)
    axes[0].set_ylabel(ylabel_for_unit(args.unit, "for_each_scaling"))
    fig.suptitle("Example: k-mers of the human genome")
    fig.tight_layout()

    # The labels that are longer than their bar go above it, in black.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, label, x, value in labels:
        if label.get_window_extent(renderer).y1 > ax.transData.transform((x, value))[1]:
            label.set_transform(mpl.transforms.offset_copy(ax.transData, fig, y=3, units="points"))
            label.set_position((x, value))
            label.set_color("black")
            label.set_bbox(None)

    out = apply_unit_scale_suffix(args.out, args.unit, args.scale, args.sink)
    if out:
        fig.savefig(out, dpi=300)
        print(f"Wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
