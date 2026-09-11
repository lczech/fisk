#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
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


def parse_case_fields(df: pd.DataFrame, column: str = "case") -> pd.DataFrame:
    """
    Split a "case" column of "key1=val1;key2=val2;..." into one new string column
    per key (e.g. "order=msb;k=17" -> columns "order"="msb", "k"="17").

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
    # "adaptive",
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
    "char_to_nt_ascii_re",
    "char_to_nt_ascii_shift",
    "char_to_nt_ifs_re",
    "char_to_nt_ifs_shift",
    "char_to_nt_switch_re",
    "char_to_nt_switch_shift",
    "char_to_nt_table_re",
    "char_to_nt_table_shift",

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
    # "adaptive",
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
    "char_to_nt_ascii_re",
    "char_to_nt_ascii_shift",
    "char_to_nt_ifs_re",
    "char_to_nt_ifs_shift",
    "char_to_nt_switch_re",
    "char_to_nt_switch_shift",
    "char_to_nt_table_re",
    "char_to_nt_table_shift",

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
    # "adaptive",
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
    "char_to_nt_ascii_re",
    "char_to_nt_ascii_shift",
    "char_to_nt_ifs_re",
    "char_to_nt_ifs_shift",
    "char_to_nt_switch_re",
    "char_to_nt_switch_shift",
    "char_to_nt_table_re",
    "char_to_nt_table_shift",

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
    "adaptive"              : "Adaptive",
    "instlatx"              : "InstLatX",
    "zp7"                   : "ZP7",

    # Extract / seq enc
    "char_to_nt_ascii_re"         : "ASCII re",
    "char_to_nt_ascii_shift"      : "ASCII shift",
    "char_to_nt_ifs_re"           : "'if' re",
    "char_to_nt_ifs_shift"        : "'if' shift",
    "char_to_nt_switch_re"        : "'switch' re",
    "char_to_nt_switch_shift"     : "'switch' shift",
    "char_to_nt_table_re"         : "Lookup table re",
    "char_to_nt_table_shift"      : "Lookup table shift",

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
    "adaptive"              : "#DB5B1F",
    "instlatx"              : "#000000",
    "zp7"                   : "#000000",

    "char_to_nt_ascii_re"       : "#3C5BBE",
    "char_to_nt_ascii_shift"    : "#3C5BBE",
    "char_to_nt_ifs_re"         : "#E9C256",
    "char_to_nt_ifs_shift"      : "#E9C256",
    "char_to_nt_switch_re"      : "#C53939",
    "char_to_nt_switch_shift"   : "#C53939",
    "char_to_nt_table_re"       : "#6AC459",
    "char_to_nt_table_shift"    : "#6AC459",


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
    "char_to_nt_ascii_re"       : "dashed",
    "char_to_nt_ascii_shift"    : "solid",
    "char_to_nt_ifs_re"         : "dashed",
    "char_to_nt_ifs_shift"      : "solid",
    "char_to_nt_switch_re"      : "dashed",
    "char_to_nt_switch_shift"   : "solid",
    "char_to_nt_table_re"       : "dashed",
    "char_to_nt_table_shift"    : "solid",
}

# Stable line order for plot consistency
BENCHMARK_ORDER = [

    # Extract / seq enc
    "char_to_nt_ifs_re",
    "char_to_nt_ifs_shift",
    "char_to_nt_switch_re",
    "char_to_nt_switch_shift",
    "char_to_nt_ascii_re",
    "char_to_nt_ascii_shift",
    "char_to_nt_table_re",
    "char_to_nt_table_shift",
    "char_to_nt_ifs",
    "char_to_nt_switch",
    "char_to_nt_ascii",
    "char_to_nt_table",

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
    "adaptive",
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
# naming scheme (internal implementation-comparison variants, not final library-facing benchmark
# names). Colors are paired by technique where the same suffix exists in both the narrow and wide
# families (_blockwise, _fixed_k, _rolling all appear twice, once per family) -- same hue, lighter
# for narrow / darker for wide -- so a technique's identity is visible across both families at a
# glance; wide-only techniques (no narrow counterpart) get their own distinct hues instead.
PACKED_KMER_VARIANT_ORDER = [
    "narrow_blockwise",
    "narrow_fixed_k",
    "narrow_aligned",
    "narrow_rolling",
    "wide_blockwise",
    "wide_fixed_k",
    "wide_aligned",
    "wide_split_k",
    "wide_hybrid",
    "wide_hoisted",
    "wide_hybrid_hoisted",
    "wide_128",
    "wide_rolling",
]

PACKED_KMER_VARIANT_COLORS = {
    "narrow_blockwise"    : "#6baed6",  # blue, light
    "wide_blockwise"      : "#08519c",  # blue, dark
    "narrow_fixed_k"      : "#74c476",  # green, light
    "wide_fixed_k"        : "#006d2c",  # green, dark
    "narrow_rolling"      : "#bdbdbd",  # grey, light
    "wide_rolling"        : "#636363",  # grey, dark
    "wide_hybrid"         : "#e6550d",  # orange
    "wide_hoisted"        : "#756bb1",  # purple
    "wide_hybrid_hoisted" : "#e377c2",  # pink
    "wide_128"            : "#8c564b",  # brown
    "narrow_aligned"      : "#9edae5",  # cyan, light
    "wide_aligned"        : "#17becf",  # cyan, dark
    "wide_split_k"        : "#bcbd22",  # olive
}
