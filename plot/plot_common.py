#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def platform_from_csv_path(csv_path: str) -> str:
    p = Path(csv_path)
    # "last part of the directory path name"
    # For ".../<CPU>/<file>.csv" this returns "<CPU>"
    return p.parent.name


# Select which benchmarks to plot at all.
# This list is across all types of benchmarks we run;
# not all are thus present in all tables.
BENCHMARKS_KEEP = {
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
}

BENCHMARK_RENAMES = {
    # Extract / seq enc
    "char_to_nt_ascii_re"         : "ascii_re",
    "char_to_nt_ascii_shift"      : "ascii_shift",
    "char_to_nt_ifs_re"           : "ifs_re",
    "char_to_nt_ifs_shift"        : "ifs_shift",
    "char_to_nt_switch_re"        : "switch_re",
    "char_to_nt_switch_shift"     : "switch_shift",
    "char_to_nt_table_re"         : "table_re",
    "char_to_nt_table_shift"      : "table_shift",
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

    # Kmer spaced, also used for bit extract order
    "naive",
    "pext",
    "bitloop",
    # "split32",
    "byte_table",
    "block_table",
    "block_table_unrolled1",
    "block_table_unrolled2",
    "block_table_unrolled4",
    "block_table_unrolled8",
    "butterfly_table",
    "adaptive",
    # "instlatx",
    # "zp7",

    "simd_pext",
    "simd_block_table_scalar",
    "simd_butterfly_table_scalar",
    "simd_block_table_sse2",
    "simd_butterfly_table_sse2",
    "simd_block_table_avx2",
    "simd_butterfly_table_avx2",
    "simd_block_table_avx512",
    "simd_butterfly_table_avx512",
    "simd_block_table_neon",
    "simd_butterfly_table_neon",
]
