#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from pathlib import Path
import matplotlib as mpl


# Hard coded font sizes across plots
mpl.rcParams.update({
    "font.size": 16, # base font
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "legend.title_fontsize": 15,
})


def platform_from_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    # "last part of the directory path name"
    # For ".../<CPU>/<file>.csv" this returns "<CPU>"
    return p.parent.name


# Canonical CPU-family / compiler orderings, shared by every cross-CPU plot script, so that
# platforms and compilers appear in the same relative order across every chart in the project
# regardless of which script produced it.
PLATFORM_ORDER = [
    "Epyc",
    "Ryzen",
    "Xeon",
    "M1",
    "M2",
    "M3",
]

COMPILER_ORDER = [
    "Clang",
    "GCC",
]


def infer_platform(raw_label: str) -> str:
    """Case-insensitive substring match of `raw_label` (e.g. a results subdirectory name like
    "AMD EPYC 9684X, Clang 17") against PLATFORM_ORDER, returning "Other" if none match."""
    s = raw_label.casefold()
    for plat in PLATFORM_ORDER:
        if plat.casefold() in s:
            return plat
    return "Other"


def infer_compiler(raw_label: str) -> str:
    """Case-insensitive substring match of `raw_label` against COMPILER_ORDER, returning "Other"
    if none match. See infer_platform() for the label format."""
    s = raw_label.casefold()
    for comp in COMPILER_ORDER:
        if comp.casefold() in s:
            return comp
    return "Other"


def platform_compiler_sort_key(raw_label: str) -> tuple[int, int, str]:
    """Sort key for a raw platform/compiler label (e.g. a results subdirectory name), ordering by
    PLATFORM_ORDER first, then COMPILER_ORDER, falling back to alphabetical for anything neither
    list recognizes. Used to keep platform axes in the same order across every cross-CPU chart,
    even in plots (like the packed k-mer comparison) that keep platform+compiler as one combined
    label rather than splitting them into separate color/hatch dimensions the way
    plot_bars_per_cpu.py does.
    """
    platform = infer_platform(raw_label)
    compiler = infer_compiler(raw_label)
    p_idx = PLATFORM_ORDER.index(platform) if platform in PLATFORM_ORDER else len(PLATFORM_ORDER)
    c_idx = COMPILER_ORDER.index(compiler) if compiler in COMPILER_ORDER else len(COMPILER_ORDER)
    return (p_idx, c_idx, raw_label)


def parse_case_fields(df: pd.DataFrame, column: str = "case") -> pd.DataFrame:
    """
    Split a "case" column of "key1=val1;key2=val2;..." into one new string column
    per key (e.g. "layout=msb;k=17" -> columns "layout"="msb", "k"="17").

    Values are left as strings; callers should convert the specific fields they
    need (e.g. `df["k"] = df["k"].astype(int)`), the same way the single-key
    "k=17"/"popcount=17"-style CSVs already do inline. A plain single-field case
    (no ";") still works, yielding just that one key's column.
    """
    df = df.copy()
    parsed = df[column].apply(
        lambda s: dict(part.split("=", 1) for part in s.split(";"))
    )
    keys = sorted({key for entry in parsed for key in entry})
    for key in keys:
        df[key] = parsed.apply(lambda entry: entry.get(key))
    return df


# Select which benchmarks to plot at all.
# This list is across all types of benchmarks we run;
# not all are thus present in all tables.

# The first list is the extended on for the early supplement benchmarks,
# where we want to show some more algorithms for comparison.
BENCHMARKS_KEEP_EXTENDED = [
    # Bit extract implementations
    "naive",
    "pext",
    "bitloop",
    # "split32",
    "byte_table",
    "block_table",
    "block_table_unrolled2",
    "block_table_unrolled4",
    "block_table_unrolled8",
    "butterfly_table",
    # "instlatx",
    # "zp7",

    # simd implementations
    "compute_spaced_kmer_hash_simd_nt_sse2",
    "compute_spaced_kmer_hash_simd_bt_sse2",
    "compute_spaced_kmer_hash_simd_nt_avx2",
    "compute_spaced_kmer_hash_simd_bt_avx2",
    "compute_spaced_kmer_hash_simd_nt_scalar",
    "compute_spaced_kmer_hash_simd_bt_scalar",

    # Extract / seq enc
    "ascii_re",
    "ascii_shift",
    "ifs_re",
    "ifs_shift",
    "switch_re",
    "switch_shift",
    "table_re",
    "table_shift",

    # Kmer spaced
    "simd_butterfly_table_sse2",
    "simd_block_table_sse2",
    "simd_butterfly_table_avx2",
    "simd_block_table_avx2",
    "simd_butterfly_table_avx512",
    "simd_block_table_avx512",
    "simd_butterfly_table_neon",
    "simd_block_table_neon",
    "simd_pext",
    "simd_butterfly_table_scalar",
    "simd_block_table_scalar",
]

# The second list is the condensed list of "good" algorithms and approaches,
# used for the supplement, where we want to show some more detail
BENCHMARKS_KEEP = [
    # Bit extract implementations
    "naive",
    "pext",
    # "bitloop",
    # "split32",
    # "byte_table",
    "block_table",
    "block_table_unrolled2",
    "block_table_unrolled4",
    "block_table_unrolled8",
    "butterfly_table",
    # "instlatx",
    # "zp7",

    # simd implementations
    "compute_spaced_kmer_hash_simd_nt_sse2",
    "compute_spaced_kmer_hash_simd_bt_sse2",
    "compute_spaced_kmer_hash_simd_nt_avx2",
    "compute_spaced_kmer_hash_simd_bt_avx2",
    # "compute_spaced_kmer_hash_simd_nt_scalar",
    # "compute_spaced_kmer_hash_simd_bt_scalar",

    # Extract / seq enc
    "ascii_re",
    "ascii_shift",
    "ifs_re",
    "ifs_shift",
    "switch_re",
    "switch_shift",
    "table_re",
    "table_shift",

    # Kmer spaced
    "simd_butterfly_table_sse2",
    "simd_block_table_sse2",
    "simd_butterfly_table_avx2",
    "simd_block_table_avx2",
    "simd_butterfly_table_avx512",
    "simd_block_table_avx512",
    "simd_butterfly_table_neon",
    "simd_block_table_neon",
    # "simd_pext",
    # "simd_butterfly_table_scalar",
    # "simd_block_table_scalar",

    # DuoHash
    "naive",
    "FSH",
    "ISSH",
    "naive",
    "FSH",
    "ISSH",
    "FSH_multi",
    "MISSH_v1",
    "MISSH_col",
    "MISSH_col_parallel",
    "MISSH_row",
]


# The third list is the small list of "recommended" algorithms and approaches,
# for the main manuscript.
BENCHMARKS_KEEP_REDUCED = [
    # Bit extract implementations
    "naive",
    "pext",
    # "bitloop",
    # "split32",
    # "byte_table",
    # "block_table",
    # "block_table_unrolled2",
    # "block_table_unrolled4",
    # "block_table_unrolled8",
    "butterfly_table",
    # "instlatx",
    # "zp7",

    # simd implementations
    "compute_spaced_kmer_hash_simd_nt_sse2",
    "compute_spaced_kmer_hash_simd_bt_sse2",
    "compute_spaced_kmer_hash_simd_nt_avx2",
    "compute_spaced_kmer_hash_simd_bt_avx2",
    "compute_spaced_kmer_hash_simd_nt_scalar",
    "compute_spaced_kmer_hash_simd_bt_scalar",

    # Extract / seq enc
    "ascii_re",
    "ascii_shift",
    "ifs_re",
    "ifs_shift",
    "switch_re",
    "switch_shift",
    "table_re",
    "table_shift",

    # Kmer spaced
    "simd_butterfly_table_sse2",
    # "simd_block_table_sse2",
    "simd_butterfly_table_avx2",
    # "simd_block_table_avx2",
    "simd_butterfly_table_avx512",
    # "simd_block_table_avx512",
    "simd_butterfly_table_neon",
    # "simd_block_table_neon",
    # "simd_pext",
    # "simd_butterfly_table_scalar",
    # "simd_block_table_scalar",
]

BENCHMARK_RENAMES = {
    # Bit extract implementations
    "naive"                 : "Naive",
    "pext"                  : "PEXT",
    "bitloop"               : "Bitloop",
    "split32"               : "Split32",
    "byte_table"            : "Byte Table",
    "block_table"           : "Block Table",
    "block_table_unrolled2" : "Block Table (2x unrolled)",
    "block_table_unrolled4" : "Block Table (4x unrolled)",
    "block_table_unrolled8" : "Block Table (8x unrolled)",
    "butterfly_table"       : "Butterfly Table",
    "instlatx"              : "InstLatX",
    "zp7"                   : "ZP7",

    # Extract / seq enc
    "ascii_re"         : "ASCII re",
    "ascii_shift"      : "ASCII shift",
    "ifs_re"           : "'if' re",
    "ifs_shift"        : "'if' shift",
    "switch_re"        : "'switch' re",
    "switch_shift"     : "'switch' shift",
    "table_re"         : "Lookup table re",
    "table_shift"      : "Lookup table shift",

    # Kmer spaced
    "simd_butterfly_table_sse2"   : "SIMD Butterfly Table SSE2",
    "simd_block_table_sse2"       : "SIMD Block Table SSE2",
    "simd_butterfly_table_avx2"   : "SIMD Butterfly Table AVX2",
    "simd_block_table_avx2"       : "SIMD Block Table AVX2",
    "simd_butterfly_table_avx512" : "SIMD Butterfly Table AVX512",
    "simd_block_table_avx512"     : "SIMD Block Table AVX512",
    "simd_butterfly_table_neon"   : "SIMD Butterfly Table Neon",
    "simd_block_table_neon"       : "SIMD Block Table Neon",
    "simd_pext"                   : "SIMD PEXT",
    "simd_butterfly_table_scalar" : "SIMD Butterfly Table scalar",
    "simd_block_table_scalar"     : "SIMD Block Table scalar",

    # Kmer extract packed
    "aligned"              : "Scalar",
    "rolling"              : "Rolling",
    "simd_narrow_sse2"     : "SIMD Narrow SSE2",
    "simd_wide_sse2"       : "SIMD Wide SSE2",
    "simd_sse2"            : "SIMD SSE2",
    "simd_narrow_avx2"     : "SIMD Narrow AVX2",
    "simd_wide_avx2"       : "SIMD Wide AVX2",
    "simd_avx2"            : "SIMD AVX2",
    "simd_narrow_avx512"   : "SIMD Narrow AVX512",
    "simd_wide_avx512"     : "SIMD Wide AVX512",
    "simd_avx512"          : "SIMD AVX512",
    "simd_narrow_neon"     : "SIMD Narrow Neon",
    "simd_wide_neon"       : "SIMD Wide Neon",
    "simd_neon"            : "SIMD Neon",
}

# "Reduced" names for the main manuscript, to keep it simple.
BENCHMARK_RENAMES_REDUCED = {
    # Kmer spaced
    "naive"                 : "Naive",
    "pext"                  : "PEXT",
    "butterfly_table"       : "Butterfly Table",

    "simd_butterfly_table_sse2"   : "SIMD SSE2",
    "simd_butterfly_table_avx2"   : "SIMD AVX2",
    "simd_butterfly_table_avx512" : "SIMD AVX512",
    "simd_butterfly_table_neon"   : "SIMD Neon",

    # Kmer extract packed
    "aligned"     : "Scalar",
    "simd_sse2"   : "SIMD SSE2",
    "simd_avx2"   : "SIMD AVX2",
    "simd_avx512" : "SIMD AVX512",
    "simd_neon"   : "SIMD Neon",
}

# Stable colors for each implementation / benchmark
BENCHMARK_COLORS = {
    # Bit extract implementations
    "naive"                 : "#23628f",
    "pext"                  : "#5F5F5F",
    "bitloop"               : "#47a1e2",
    "split32"               : "#ff7f0e",
    "byte_table"            : "#904ce9",
    "block_table"           : "#a1d99b",
    "block_table_unrolled2" : "#74c476",
    "block_table_unrolled4" : "#31a354",
    "block_table_unrolled8" : "#006d2c",
    "butterfly_table"       : "#BB2F98",
    "instlatx"              : "#000000",
    "zp7"                   : "#000000",

    "ascii_re"       : "#3C5BBE",
    "ascii_shift"    : "#3C5BBE",
    "ascii_validate"      : "#3C5BBE",
    "ascii_assume_valid"  : "#3C5BBE",
    "ifs_re"         : "#E9C256",
    "ifs_shift"      : "#E9C256",
    "switch_re"      : "#C53939",
    "switch_shift"   : "#C53939",
    "table_re"       : "#6AC459",
    "table_shift"    : "#6AC459",


    # Kmer spaced
    "simd_butterfly_table_sse2"   : "#BB2F98",
    "simd_block_table_sse2"       : "#006d2c",
    "simd_butterfly_table_avx2"   : "#BB2F98",
    "simd_block_table_avx2"       : "#006d2c",
    "simd_butterfly_table_avx512" : "#BB2F98",
    "simd_block_table_avx512"     : "#006d2c",
    "simd_butterfly_table_neon"   : "#BB2F98",
    "simd_block_table_neon"       : "#006d2c",
    "simd_pext"                   : "#5F5F5F",
    "simd_butterfly_table_scalar" : "#BB2F98",
    "simd_block_table_scalar"     : "#006d2c",
}

BENCHMARK_LINESTYLES = {
    # Extract / seq enc
    "ascii_re"       : "dashed",
    "ascii_shift"    : "solid",
    "ascii_validate"      : "dashed",
    "ascii_assume_valid"  : "solid",
    "ifs_re"         : "dashed",
    "ifs_shift"      : "solid",
    "switch_re"      : "dashed",
    "switch_shift"   : "solid",
    "table_re"       : "dashed",
    "table_shift"    : "solid",
}

# Stable line order for plot consistency
BENCHMARK_ORDER = [

    # Extract / seq enc
    "ifs_re",
    "ifs_shift",
    "switch_re",
    "switch_shift",
    "ascii_re",
    "ascii_shift",
    "table_re",
    "table_shift",
    "ifs",
    "switch",
    "ascii_validate",
    "ascii_assume_valid",
    "table",

    # Kmer spaced, also used for bit extract order
    "naive",
    "pext",
    "bitloop",
    # "split32",
    "byte_table",
    "butterfly_table",
    "block_table",
    "block_table_unrolled1",
    "block_table_unrolled2",
    "block_table_unrolled4",
    "block_table_unrolled8",
    # "instlatx",
    # "zp7",

    "simd_pext",
    "simd_butterfly_table_scalar",
    "simd_block_table_scalar",
    "simd_butterfly_table_sse2",
    "simd_block_table_sse2",
    "simd_butterfly_table_avx2",
    "simd_block_table_avx2",
    "simd_butterfly_table_avx512",
    "simd_block_table_avx512",
    "simd_butterfly_table_neon",
    "simd_block_table_neon",
]

# Stable order and colors for the kmer_extract_packed benchmark family (packed.hpp), kept as its
# own dict rather than folded into BENCHMARK_COLORS/BENCHMARK_ORDER above since it's a distinct
# naming scheme. Direct narrow/wide rows are retained for code-generation comparisons; dispatchers
# are the user-facing entry points.
PACKED_KMER_VARIANT_ORDER = [
    "aligned",
    "simd_narrow_sse2",
    "simd_wide_sse2",
    "simd_sse2",
    "simd_narrow_avx2",
    "simd_wide_avx2",
    "simd_avx2",
    "simd_narrow_avx512",
    "simd_wide_avx512",
    "simd_avx512",
    "simd_narrow_neon",
    "simd_wide_neon",
    "simd_neon",
    "rolling",
]

PACKED_KMER_VARIANT_COLORS = {
    "aligned" : "#107D84",
    "rolling" : "#636363",  # grey
    # SSE2: purple shades, ordered narrow -> wide -> dispatcher.
    "simd_narrow_sse2" : "#c5b0d5",
    "simd_wide_sse2" : "#9467bd",
    "simd_sse2" : "#5C338A",
    # AVX2: green shades, ordered narrow -> wide -> dispatcher.
    "simd_narrow_avx2" : "#78c679",
    "simd_wide_avx2" : "#2ca02c",
    "simd_avx2" : "#207542",
    # AVX-512: red shades, ordered narrow -> wide -> dispatcher.
    "simd_narrow_avx512" : "#fb6a4a",
    "simd_wide_avx512" : "#d62728",
    "simd_avx512" : "#a23236",
    # NEON: blue shades, ordered narrow -> wide -> dispatcher.
    "simd_narrow_neon" : "#9ecae1",
    "simd_wide_neon" : "#6baed6",
    "simd_neon" : "#23609d",
}

# Headline subset of PACKED_KMER_VARIANT_ORDER: the scalar baseline plus each ISA's public
# dispatcher only -- no narrow/wide/rolling implementation detail. Each dispatcher already picks
# narrow or wide internally depending on k, so a chart already split by k-tier (e.g. one figure per
# narrow/wide range) loses nothing by showing only the dispatcher row: it *is* whichever of
# narrow/wide applies at that k. Used as the default (non---extended) comparison view.
PACKED_KMER_VARIANT_ORDER_REDUCED = [
    "aligned",
    "simd_sse2",
    "simd_avx2",
    "simd_avx512",
    "simd_neon",
]

# Fold the packed k-mer variant names into the generic BENCHMARKS_KEEP*/BENCHMARK_ORDER lists too,
# so plot_bars_per_cpu.py's cross-CPU summary shows them instead of filtering every row out (those
# lists predate this benchmark family and never got extended for it). Appended rather than merged
# in above since BENCHMARK_ORDER/BENCHMARKS_KEEP are defined earlier in this file, before
# PACKED_KMER_VARIANT_ORDER exists.
BENCHMARK_ORDER += PACKED_KMER_VARIANT_ORDER
BENCHMARKS_KEEP += PACKED_KMER_VARIANT_ORDER
BENCHMARKS_KEEP_EXTENDED += PACKED_KMER_VARIANT_ORDER
BENCHMARKS_KEEP_REDUCED += PACKED_KMER_VARIANT_ORDER_REDUCED


# -----------------------------------------------------------------------------
#     Display unit: throughput (default) vs legacy ns/op
# -----------------------------------------------------------------------------

# Which "thing" one unit of ns_per_op actually measures, per benchmark suite (see each suite's
# .units_fn() in benchmarks/*/bench.cpp) -- used to build a correct throughput axis label. Suites
# not listed here (e.g. an old CSV from a removed suite, or a script that combines multiple
# suites) fall back to the generic "ops" noun in unit_noun_for_suite() below.
SUITE_UNIT_NOUN = {
    "kmer_extract"        : "k-mers",
    "kmer_extract_packed" : "k-mers",
    "kmer_clark"          : "k-mers",
    "kmer_spaced_single"  : "k-mers",
    "kmer_spaced_multi"   : "k-mers",
    "char_encoder"        : "bases",
    "seq_pack"            : "bases",
    "bit_extract_weights" : "ops",
    "bit_extract_blocks"  : "ops",
}

# Fixed divisor for throughput display: always giga-<unit>/s, even for the couple of suites whose
# peak throughput doesn't quite reach 1e9, so every plot in the paper reads on the same scale.
THROUGHPUT_SCALE = 1e9
THROUGHPUT_PREFIX = "G"


def unit_noun_for_suite(suite: str | None) -> str:
    """Return the throughput noun (e.g. "k-mers", "bases", "ops") for a suite name, falling back
    to the generic "ops" for anything not in SUITE_UNIT_NOUN (unknown/missing suite)."""
    if suite is None:
        return "ops"
    return SUITE_UNIT_NOUN.get(suite, "ops")


def ns_per_op_to_throughput(ns_per_op):
    """Convert ns/op (scalar, Series, or ndarray) to throughput in THROUGHPUT_PREFIX-units/s."""
    return (1e9 / ns_per_op) / THROUGHPUT_SCALE


def convert_for_display(ns_per_op, unit: str):
    """Convert a ns_per_op scalar/Series/ndarray to the requested --unit ("ns": identity, "ops":
    throughput via ns_per_op_to_throughput()). Aggregation (mean/min/max) must already be done in
    ns/op space before calling this -- inverting first and aggregating after is a different,
    non-equivalent statistic (harmonic- vs. arithmetic-mean-like behavior)."""
    if unit == "ns":
        return ns_per_op
    return ns_per_op_to_throughput(ns_per_op)


def ylabel_for_unit(unit: str, suite: str | None = None) -> str:
    """Y-axis label for the selected --unit: "Time per operation [ns]" for "ns", or a suite-aware
    "Throughput [G <noun>/s]" for "ops"."""
    if unit == "ns":
        return "Time per operation [ns]"
    noun = unit_noun_for_suite(suite)
    return f"Throughput [{THROUGHPUT_PREFIX} {noun}/s]"


def compute_axis_limits(values, scale: str, headroom: float = 1.05) -> tuple[float, float]:
    """
    Data-driven (y_min, y_max) for an axis showing `values` (already converted to display units)
    on the given `scale` ("linear" or "log"):
    - linear: floor at 0, ceiling at max(values) * headroom.
    - log: floor and ceiling both get the same proportional headroom below/above the data's real
      (positive) min/max. The limits themselves need not land on a round power of ten --
      Matplotlib's log tick locator (see apply_yscale()) still places major ticks at the enclosing
      powers of ten regardless, leaving a little blank space near the axis edges.
    Non-finite values are dropped (and non-positive ones too, for "log", since they're undefined
    on a log axis); an all-filtered/empty input falls back to a fixed placeholder range rather
    than raising, so a caller can still render an (empty) plot instead of crashing.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if scale == "log":
        arr = arr[arr > 0]

    if arr.size == 0:
        return (0.0, 1.0) if scale == "linear" else (0.1, 1.0)

    vmax = float(arr.max())
    if scale == "linear":
        return 0.0, vmax * headroom

    vmin = float(arr.min())
    return vmin / headroom, vmax * headroom


def _log_tick_formatter(value: float, _pos=None) -> str:
    """Render a log-axis tick as a plain decimal (e.g. "0.1", "10") instead of Matplotlib's
    default scientific "10^-1" style -- values here are already pre-scaled by THROUGHPUT_SCALE, so
    the natural range is small, easy-to-read decimals/integers rather than large exponents."""
    if value == 0:
        return "0"
    return f"{value:g}"


def apply_yscale(ax, scale: str) -> None:
    """Apply the requested y-axis scale ("linear" or "log") to `ax`. For "log", major ticks land
    at 1/2/5 times each power of ten (not just bare decades) with a plain-decimal formatter (e.g.
    "0.2", "1", "5" rather than "10^-1"/only "1","10") -- most plots here span less than 2 decades,
    where a bare-decade locator would often produce only zero or one labeled tick."""
    if scale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(3, 4, 6, 7, 8, 9)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_yscale("linear")


def add_unit_scale_args(parser: argparse.ArgumentParser) -> None:
    """Add the --unit/--scale/--y-max/--y-min flags shared by every ns_per_op plotting script, so
    names/defaults/help text can't drift between scripts."""
    parser.add_argument(
        "--unit", choices=["ops", "ns"], default="ops",
        help='Display unit: "ops" for throughput (default), "ns" for legacy time-per-operation.',
    )
    parser.add_argument(
        "--scale", choices=["log", "linear"], default="log",
        help="Y-axis scale (default: log).",
    )
    parser.add_argument(
        "--y-max", type=float, default=None,
        help="Fixed y-axis upper limit, in the selected --unit (default: auto-scaled from the data).",
    )
    parser.add_argument(
        "--y-min", type=float, default=None,
        help="Fixed y-axis lower limit, in the selected --unit (default: 0 for --scale linear, "
             "auto-scaled from the data for --scale log).",
    )


def apply_unit_scale_suffix(path: str | None, unit: str, scale: str) -> str | None:
    """Insert an unconditional "_<unit>_<scale>" suffix before a path's extension (e.g.
    "foo.png" -> "foo_ops_log.png"), so runs with different --unit/--scale never silently
    overwrite each other's output. No-op for path=None (interactive display)."""
    if path is None:
        return None
    root, ext = os.path.splitext(path)
    return f"{root}_{unit}_{scale}{ext}"
BENCHMARK_COLORS.update(PACKED_KMER_VARIANT_COLORS)
