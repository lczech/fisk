#!/usr/bin/env python3
"""
Chunked extraction from ASCII: the direct loops on the text, the chunked loops, and packing the
whole sequence first, side by side, one panel per CPU/compiler.

Reads, from each CPU/compiler directory, the tables of the extraction from ASCII, of the packing,
and of the extraction from packed sequences, and draws a grid of grouped bar charts: x-axis = the
kind of input (contiguous or spaced k-mers with a single mask, unchecked or checked), three bars
each, in ns per position:
 - direct: the fastest variant of the extraction from ASCII that works on the text itself, such
   as rolling_scalar_table or rolling_simd_butterfly_avx2_scalars; there is none without checks;
 - chunked: the fastest chunked variant of that table;
 - packed: the fastest variant of sequence_pack_unchecked or sequence_pack_checked, plus the
   fastest one of the extraction from packed sequences, that is, packing the whole sequence
   first, and then extracting from it, timed separately.
Each variant is first averaged over the cases of its table: the values of k, the masks, and the
layouts, with the ACGT encoding only, and without the external references of the packing. The
spaced tables are restricted to MSB, see
normalize_spaced_layout(). The SIMD variants of the extraction count with their single k-mers
(`_scalars`), as the chunked ones only come in that form; those that call back with whole vectors
are left out.

Usage example:
  python plot/plot_chunked_per_cpu.py \
      --dir "results/Intel Core 7 240H, Clang 21" --dir "results/Intel Core 7 240H, GCC 15" \
      --out results/Summaries/chunked_per_cpu.png

If --out is omitted, shows interactively.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, '.')
from plot_common import *


FORMS = ["direct", "chunked", "packed"]

FORM_LABELS = {
    "direct": "direct, on the text",
    "chunked": "chunked",
    "packed": "pack, then extract",
}

FORM_COLORS = {
    "direct": "#C44E52",
    "chunked": "#4C72B0",
    "packed": "#55A868",
}

# The kinds of input, in plot order: the table of the extraction from ASCII, of the packing, and
# of the extraction from packed sequences.
INPUTS = [
    ("k-mers, unchecked", "kmer_extract_ascii_unchecked", "sequence_pack_unchecked",
     "kmer_extract_packed_unchecked"),
    ("k-mers, checked", "kmer_extract_ascii_checked", "sequence_pack_checked",
     "kmer_extract_packed_checked"),
    ("spaced k-mers, unchecked", "spaced_kmer_extract_ascii_unchecked_single",
     "sequence_pack_unchecked", "spaced_kmer_extract_packed_unchecked_single"),
    ("spaced k-mers, checked", "spaced_kmer_extract_ascii_checked_single",
     "sequence_pack_checked", "spaced_kmer_extract_packed_checked_single"),
]

# The variants of the tables of the extraction from ASCII that are not direct loops on the text.
NOT_DIRECT_PREFIXES = ("chunked_", "pack_then_", "naive", "missh")


def load_means(directory, name):
    """
    The mean ns per op of each variant of the table `name` in `directory`, over its cases with
    the ACGT encoding (where the case has one), and for the spaced tables with MSB; None if the
    table is missing.
    """
    path = os.path.join(directory, name + ".csv")
    if not os.path.isfile(path):
        return None
    df = parse_case_fields(normalize_spaced_layout(pd.read_csv(path)))
    if "encoding" in df.columns:
        df = df[df["encoding"].isna() | (df["encoding"] == "acgt")]
    # The SIMD variants that call back with whole vectors: `_vectors` in the spaced suites, and
    # without a suffix in the others, whose single k-mers are `_scalars`.
    simd = df["benchmark"].str.contains("_simd")
    df = df[~simd | df["benchmark"].str.endswith("_scalars")]
    df = df[~df["benchmark"].str.startswith(EXTERNAL_PREFIXES)]
    return df.groupby("benchmark")["ns_per_op"].mean()


def load_pivot(directory, unit):
    """
    Return ("kmer_extract_ascii_checked", pivot) for one directory, with pivot index = kind of
    input, columns = FORMS, values in the display unit; a form that a directory lacks is NaN.
    """
    rows = {}
    for label, ascii_name, pack_name, packed_name in INPUTS:
        ascii_means = load_means(directory, ascii_name)
        if ascii_means is None:
            continue
        is_chunked = ascii_means.index.str.startswith("chunked_")
        is_direct = ~ascii_means.index.str.startswith(NOT_DIRECT_PREFIXES)
        pack_means = load_means(directory, pack_name)
        packed_means = load_means(directory, packed_name)
        rows[label] = {
            "direct": ascii_means[is_direct].min() if is_direct.any() else float("nan"),
            "chunked": ascii_means[is_chunked].min() if is_chunked.any() else float("nan"),
            "packed": (
                pack_means.min() + packed_means.min()
                if pack_means is not None and packed_means is not None else float("nan")
            ),
        }
    pivot = pd.DataFrame.from_dict(rows, orient="index", columns=FORMS)
    return "kmer_extract_ascii_checked", convert_for_display(pivot, unit)


def main():
    ap = argparse.ArgumentParser(
        description="Direct, chunked, and pack-then-extract, per kind of input, one panel per CPU."
    )
    ap.add_argument(
        "--dir", action="append", default=[],
        help="CPU/compiler directory with the tables (repeatable).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output image path (if omitted, show interactively).",
    )
    ap.add_argument("--title", default=None, help="Figure title override.")
    # Accepted for plot_modes() in plot_all_cpus.sh; the fastest variants do not depend on it.
    add_mode_arg(ap)
    add_unit_scale_args(ap)
    args = ap.parse_args()

    # A directory without any table of the chunked extraction is left out. Not an error: expected
    # while the chunked variants have not been run on a CPU yet.
    dirs = [
        d for d in args.dir
        if any(os.path.isfile(os.path.join(d, a + ".csv")) for _, a, _, _ in INPUTS)
    ]
    if not dirs:
        warn("Nothing to plot: no directory has a table of the extraction from ASCII")
        return

    # plot_axis_grid() takes the label of each panel from the directory of its first path.
    paths = [os.path.join(d, INPUTS[0][1] + ".csv") for d in dirs]
    out = apply_unit_scale_suffix(args.out, args.unit, args.scale)
    plot_axis_grid(
        paths,
        lambda path, unit: load_pivot(os.path.dirname(path), unit),
        FORM_LABELS, FORM_COLORS,
        "extraction from ASCII: direct, chunked, and pack then extract",
        out, args.unit, args.scale, args.y_min, args.y_max,
        args.title or "K-mer extraction from ASCII: direct, chunked, and pack then extract",
    )


if __name__ == "__main__":
    main()
