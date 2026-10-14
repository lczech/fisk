#!/usr/bin/env python3
"""Sequence packing: four overview plots and two focused FastDNA comparisons.

Each encoding/layout gets its own file. Y-limits are shared within each view set.
Use --summary for the existing CPU/compiler grouped-bar style across CSV files.

Examples:
  python plot/plot_seq_pack.py results/seq_pack.csv --out-dir results
  python plot/plot_seq_pack.py --glob 'results/*/seq_pack.csv' --summary \
      --out-dir results/Summaries
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_bars_per_cpu import main as plot_platform_bars
from plot_case_summary import make_impl_summary_plot
from plot_common import (
    SEQ_PACK_COLORS,
    SEQ_PACK_FOCUSED,
    SEQ_PACK_ORDER,
    add_unit_scale_args,
    apply_unit_scale_suffix,
    compute_axis_limits,
    convert_for_display,
    infer_compiler,
    infer_platform,
    parse_case_fields,
    platform_from_csv_path,
)


# -------------------------------------------------------------------------------------------------
#     Data and View Selection
# -------------------------------------------------------------------------------------------------

COMBINATIONS = [("acgt", "msb"), ("acgt", "lsb"), ("actg", "msb"), ("actg", "lsb")]


def load_results(csv_path):
    """Load packing rows without averaging different output conventions."""
    df = pd.read_csv(csv_path)
    required = {"suite", "case", "benchmark", "ns_per_op"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    df = df[df["suite"] == "seq_pack"].copy()
    if df.empty:
        raise ValueError(f"{csv_path}: no seq_pack rows")
    df = parse_case_fields(df)
    if not {"encoding", "layout"}.issubset(df.columns):
        raise ValueError(f"{csv_path}: cases must specify encoding and layout")
    valid_cases = df["encoding"].isin(["acgt", "actg"]) & df["layout"].isin(["msb", "lsb"])
    if not valid_cases.all():
        raise ValueError(f"{csv_path}: unsupported encoding/layout case")
    df["ns_per_op"] = pd.to_numeric(df["ns_per_op"], errors="raise")
    if not (np.isfinite(df["ns_per_op"]) & (df["ns_per_op"] > 0)).all():
        raise ValueError(f"{csv_path}: ns_per_op must be finite and positive")
    label = platform_from_csv_path(csv_path)
    df["platform"] = infer_platform(label)
    df["compiler"] = infer_compiler(label)
    return df


def select_views(df):
    """Keep every overview variant; only show a focused view with reference measurements."""
    views = {}
    for encoding, layout in COMBINATIONS:
        rows = df[(df["encoding"] == encoding) & (df["layout"] == layout)]
        if not rows.empty:
            views[("overview", encoding, layout)] = rows
        if encoding != "actg":
            continue
        focused = rows[rows["benchmark"].isin(SEQ_PACK_FOCUSED)]
        if focused["benchmark"].str.startswith("fastdna_").any():
            views[("fastdna", encoding, layout)] = focused
        else:
            print(f"Skipped focused {encoding}/{layout}: no FastDNA measurements")
    return views


def benchmark_order(rows):
    """Use the curated order, retaining unlisted variants in the full overview."""
    present = set(rows["benchmark"])
    return [b for b in SEQ_PACK_ORDER if b in present] + sorted(present - set(SEQ_PACK_ORDER))


def shared_limits(views, args):
    """Compute one y-range per view set, from the values actually plotted."""
    limits = {}
    for view in ["overview", "fastdna"]:
        selected = [rows for key, rows in views.items() if key[0] == view]
        if not selected:
            continue
        rows = pd.concat(selected, ignore_index=True)
        means = rows.groupby(["encoding", "layout", "benchmark", "platform", "compiler"])[
            "ns_per_op"
        ].mean()
        y_min, y_max = compute_axis_limits(convert_for_display(means, args.unit), args.scale)
        # Reserve room for annotations and the summary renderer's existing bar ceiling.
        y_max *= 1.3 if args.scale == "log" else 1.2
        limits[view] = (
            args.y_min if args.y_min is not None else y_min,
            args.y_max if args.y_max is not None else y_max,
        )
    return limits


# -------------------------------------------------------------------------------------------------
#     Rendering
# -------------------------------------------------------------------------------------------------

def plot_views(files, views, limits, args):
    """Reuse the per-CPU and cross-CPU bar renderers for each fixed output convention."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (view, encoding, layout), rows in views.items():
        order = benchmark_order(rows)
        y_min, y_max = limits[view]
        stem = f"seq_pack_{view}_{encoding}_{layout}"
        suffix = "_per_cpu" if args.summary else ""
        out = str(out_dir / f"{stem}{suffix}.{args.format}")
        heading = "All sequence-packing implementations" if view == "overview" else "FastDNA references vs fisk"
        title = f"{heading}: {encoding.upper()} / {layout.upper()}"
        if args.summary:
            argv = [
                "--legend-outside",
                "--suite", "seq_pack", "--case-filter", f"encoding={encoding}",
                "--case-filter", f"layout={layout}", "--unit", args.unit, "--scale", args.scale,
                "--y-min", str(y_min), "--y-max", str(y_max), "--title", title, "--out", out,
            ]
            for path in files:
                argv.extend(["--file", path])
            for name in order:
                argv.extend(["--benchmark", name])
            plot_platform_bars(argv)
        else:
            title = f"{platform_from_csv_path(files[0])}\n{title}"
            make_impl_summary_plot(
                rows, "seq_pack", title, apply_unit_scale_suffix(out, args.unit, args.scale),
                benchmark_order=order, benchmark_colors=SEQ_PACK_COLORS,
                unit=args.unit, scale=args.scale, y_min=y_min, y_max=y_max, show_range=False,
            )
            plt.close("all")


# -------------------------------------------------------------------------------------------------
#     Command Line
# -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="Sequence-packing CSV for a single CPU/compiler")
    parser.add_argument("--file", action="append", default=[], help="Input CSV (repeatable)")
    parser.add_argument("--glob", action="append", default=[], help="Input glob (repeatable)")
    parser.add_argument("--summary", action="store_true", help="Group bars by CPU/compiler across inputs")
    parser.add_argument("--out-dir", required=True, help="Directory for the six view files")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    add_unit_scale_args(parser)
    args = parser.parse_args(argv)
    files = ([args.csv] if args.csv else []) + args.file
    for pattern in args.glob:
        files.extend(sorted(glob.glob(pattern)))
    files = list(dict.fromkeys(files))
    if not files:
        parser.error("Provide a CSV, --file, or --glob")
    if not args.summary and len(files) != 1:
        parser.error("Multiple inputs require --summary")
    df = pd.concat([load_results(path) for path in files], ignore_index=True)
    if "sink" in df.columns and df["sink"].nunique() > 1:
        parser.error("Inputs use different benchmark sink modes")
    views = select_views(df)
    plot_views(files, views, shared_limits(views, args), args)


if __name__ == "__main__":
    main()
