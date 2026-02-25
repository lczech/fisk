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
    # PEXT implementations
    "pext_hw_bmi2",
    "pext_sw_bitloop",
    # "pext_sw_split32",
    "pext_sw_table8",
    "pext_sw_adaptive",
    "pext_sw_block_table",
    "pext_sw_block_table_unrolled2",
    "pext_sw_block_table_unrolled4",
    "pext_sw_block_table_unrolled8",
    # "pext_sw_instlatx",
    # "pext_sw_zp7",

    # Extract / seq enc
    # "char_to_nt_ascii_nothrow_re",
    # "char_to_nt_ascii_nothrow_shift",
    "char_to_nt_ascii_throw_re",
    "char_to_nt_ascii_throw_shift",
    # "char_to_nt_ifs_nothrow_re",
    # "char_to_nt_ifs_nothrow_shift",
    "char_to_nt_ifs_throw_re",
    "char_to_nt_ifs_throw_shift",
    # "char_to_nt_switch_nothrow_re",
    # "char_to_nt_switch_nothrow_shift",
    "char_to_nt_switch_throw_re",
    "char_to_nt_switch_throw_shift",
    # "char_to_nt_table_nothrow_re",
    # "char_to_nt_table_nothrow_shift",
    "char_to_nt_table_throw_re",
    "char_to_nt_table_throw_shift",
}

BENCHMARK_RENAMES = {
    # Extract / seq enc
    "char_to_nt_ascii_nothrow_re"       : "ascii_re",
    "char_to_nt_ascii_nothrow_shift"    : "ascii_shift",
    "char_to_nt_ascii_throw_re"         : "ascii_re",
    "char_to_nt_ascii_throw_shift"      : "ascii_shift",
    "char_to_nt_ifs_nothrow_re"         : "ifs_re",
    "char_to_nt_ifs_nothrow_shift"      : "ifs_shift",
    "char_to_nt_ifs_throw_re"           : "ifs_re",
    "char_to_nt_ifs_throw_shift"        : "ifs_shift",
    "char_to_nt_switch_nothrow_re"      : "switch_re",
    "char_to_nt_switch_nothrow_shift"   : "switch_shift",
    "char_to_nt_switch_throw_re"        : "switch_re",
    "char_to_nt_switch_throw_shift"     : "switch_shift",
    "char_to_nt_table_nothrow_re"       : "table_re",
    "char_to_nt_table_nothrow_shift"    : "table_shift",
    "char_to_nt_table_throw_re"         : "table_re",
    "char_to_nt_table_throw_shift"      : "table_shift",
}

# Stable colors for each implementation / benchmark
BENCHMARK_COLORS = {
    # PEXT implementations
    "pext_hw_bmi2":        "#5F5F5F",
    "pext_sw_bitloop":     "#47a1e2",
    "pext_sw_split32":     "#ff7f0e",
    "pext_sw_table8":      "#884ed3",
    "pext_sw_adaptive":    "#D35820",
    "pext_sw_block_table":           "#a1d99b",
    "pext_sw_block_table_unrolled2": "#74c476",
    "pext_sw_block_table_unrolled4": "#31a354",
    "pext_sw_block_table_unrolled8": "#006d2c",
    "pext_sw_instlatx":    "#000000",
    "pext_sw_zp7":         "#000000",

    "char_to_nt_switch_throw_re"        : "#C00000",
    "char_to_nt_switch_nothrow_re"      : "#C00000",
    "char_to_nt_ifs_throw_re"           : "#F0B512",
    "char_to_nt_ifs_nothrow_re"         : "#F0B512",
    "char_to_nt_ascii_throw_re"         : "#1037B9",
    "char_to_nt_ascii_nothrow_re"       : "#1037B9",
    "char_to_nt_table_throw_re"         : "#36C21A",
    "char_to_nt_table_nothrow_re"       : "#36C21A",
    "char_to_nt_switch_throw_shift"     : "#C00000",
    "char_to_nt_switch_nothrow_shift"   : "#C00000",
    "char_to_nt_ifs_throw_shift"        : "#F0B512",
    "char_to_nt_ifs_nothrow_shift"      : "#F0B512",
    "char_to_nt_ascii_throw_shift"      : "#1037B9",
    "char_to_nt_ascii_nothrow_shift"    : "#1037B9",
    "char_to_nt_table_throw_shift"      : "#36C21A",
    "char_to_nt_table_nothrow_shift"    : "#36C21A",
}

BENCHMARK_LINESTYLES = {
    # Extract / seq enc
    "char_to_nt_switch_throw_re"        : "dotted",
    "char_to_nt_switch_nothrow_re"      : "dotted",
    "char_to_nt_ifs_throw_re"           : "dotted",
    "char_to_nt_ifs_nothrow_re"         : "dotted",
    "char_to_nt_ascii_throw_re"         : "dotted",
    "char_to_nt_ascii_nothrow_re"       : "dotted",
    "char_to_nt_table_throw_re"         : "dotted",
    "char_to_nt_table_nothrow_re"       : "dotted",
    "char_to_nt_switch_throw_shift"     : "solid",
    "char_to_nt_switch_nothrow_shift"   : "solid",
    "char_to_nt_ifs_throw_shift"        : "solid",
    "char_to_nt_ifs_nothrow_shift"      : "solid",
    "char_to_nt_ascii_throw_shift"      : "solid",
    "char_to_nt_ascii_nothrow_shift"    : "solid",
    "char_to_nt_table_throw_shift"      : "solid",
    "char_to_nt_table_nothrow_shift"    : "solid",
}

# Stable line order for plot consistency
BENCHMARK_ORDER = [
    # PEXT implementations
    "pext_hw_bmi2",
    "pext_sw_bitloop",
    "pext_sw_split32",
    "pext_sw_table8",
    "pext_sw_adaptive",
    "pext_sw_block_table",
    "pext_sw_block_table_unrolled2",
    "pext_sw_block_table_unrolled4",
    "pext_sw_block_table_unrolled8",
    "pext_sw_instlatx",
    "pext_sw_zp7",

    # Extract / seq enc
    "char_to_nt_switch_throw_re",
    "char_to_nt_switch_nothrow_re",
    "char_to_nt_ifs_throw_re",
    "char_to_nt_ifs_nothrow_re",
    "char_to_nt_ascii_throw_re",
    "char_to_nt_ascii_nothrow_re",
    "char_to_nt_table_throw_re",
    "char_to_nt_table_nothrow_re",
    "char_to_nt_switch_throw_shift",
    "char_to_nt_switch_nothrow_shift",
    "char_to_nt_ifs_throw_shift",
    "char_to_nt_ifs_nothrow_shift",
    "char_to_nt_ascii_throw_shift",
    "char_to_nt_ascii_nothrow_shift",
    "char_to_nt_table_throw_shift",
    "char_to_nt_table_nothrow_shift",
]
