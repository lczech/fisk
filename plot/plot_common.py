#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import re
import sys
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
    "Core 7",
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


# Messages on stderr, as in plot/log.sh for the shell scripts: warn() for something that is left out
# of a plot, such as a result table without the rows that the plot needs. The prefix is colored when
# stderr is a terminal, and plain in logs.
_WARN_PREFIX = "\033[1;33mWARNING:\033[0m" if sys.stderr.isatty() else "WARNING:"


def warn(message: str) -> None:
    print(f"{_WARN_PREFIX} {message}", file=sys.stderr)


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


def normalize_spaced_layout(df: pd.DataFrame) -> pd.DataFrame:
    """
    The spaced k-mer benchmarks run in both Layouts, with the Layout in front of the mask in their
    case, as "layout=msb;mask=0". Keep the MSB rows, with the case of the mask only, and drop the
    LSB ones, which only serve to check that both layouts perform the same.
    """
    # As strings: pandas reads a case such as "n/a" (kmer_clark) as a missing value.
    spaced = df["suite"].astype(str).str.startswith("spaced_kmer_extract")
    df = df[~(spaced & df["case"].astype(str).str.startswith("layout=lsb;"))].copy()
    spaced = df["suite"].astype(str).str.startswith("spaced_kmer_extract")
    if spaced.any():
        df.loc[spaced, "case"] = df.loc[spaced, "case"].str.removeprefix("layout=msb;")
    return df


# -------------------------------------------------------------------------------------------------
#     Plot Modes
# -------------------------------------------------------------------------------------------------

# Each plot comes in up to three modes, from the hand-picked variants of the manuscript to all of
# them. The lists are across all suites, so not all of their variants are in every table.
#  - "main": the few recommended algorithms and approaches, for the main manuscript, with the short
#    labels of BENCHMARK_RENAMES_MAIN;
#  - "supp": the variants for the supplement, with more algorithms for comparison and more detail;
#  - "full": every variant of the tables, unfiltered, to see what there is.
# The plot scripts take --mode, see add_mode_arg(), and the plot-all scripts name their outputs
# after it, as <name>_<mode>.
PLOT_MODES = ["main", "supp", "full"]

# The variants of the supplement.
BENCHMARKS_SUPP = [
    # Bit extract implementations
    "naive",
    "pext",
    "rolling_scalar_pext",
    "bytewise_scalar_pext",
    "pext_scalar",
    "bitloop",
    "rolling_scalar_bitloop",
    "bytewise_scalar_bitloop",
    # "split32",
    "byte_table",
    "rolling_scalar_byte_table",
    "bytewise_scalar_byte_table",
    "block_table",
    "rolling_scalar_block_table",
    "bytewise_scalar_block_table",
    "block_table_unrolled2",
    "rolling_scalar_block_table_unrolled2",
    "bytewise_scalar_block_table_unrolled2",
    "block_table_unrolled4",
    "rolling_scalar_block_table_unrolled4",
    "bytewise_scalar_block_table_unrolled4",
    "block_table_unrolled8",
    "rolling_scalar_block_table_unrolled8",
    "bytewise_scalar_block_table_unrolled8",
    "butterfly_table",
    "rolling_scalar_butterfly_table",
    "bytewise_scalar_butterfly_table",
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
    "reextract_scalar_ascii",
    "rolling_scalar_ascii",
    "reextract_scalar_ifs",
    "rolling_scalar_ifs",
    "reextract_scalar_switch",
    "rolling_scalar_switch",
    "reextract_scalar_table",
    "rolling_scalar_table",
    "pack_then_bytewise_scalar",
    "rolling_avx2",
    "rolling_swar",
    "chunked_scalar_threadlocal",
    "chunked_scalar_scratch",

    # Kmer spaced, from text and from packed sequences
    "rolling_simd_butterfly_table_sse2_vectors",
    "bytewise_simd_butterfly_table_sse2_vectors",
    "rolling_simd_butterfly_table_sse2_scalars",
    "bytewise_simd_butterfly_table_sse2_scalars",
    "rolling_simd_block_table_sse2_vectors",
    "bytewise_simd_block_table_sse2_vectors",
    "rolling_simd_block_table_sse2_scalars",
    "bytewise_simd_block_table_sse2_scalars",
    "rolling_simd_butterfly_table_avx2_vectors",
    "bytewise_simd_butterfly_table_avx2_vectors",
    "rolling_simd_butterfly_table_avx2_scalars",
    "bytewise_simd_butterfly_table_avx2_scalars",
    "rolling_simd_block_table_avx2_vectors",
    "bytewise_simd_block_table_avx2_vectors",
    "rolling_simd_block_table_avx2_scalars",
    "bytewise_simd_block_table_avx2_scalars",
    "rolling_simd_butterfly_table_avx512_vectors",
    "bytewise_simd_butterfly_table_avx512_vectors",
    "rolling_simd_butterfly_table_avx512_scalars",
    "bytewise_simd_butterfly_table_avx512_scalars",
    "rolling_simd_block_table_avx512_vectors",
    "bytewise_simd_block_table_avx512_vectors",
    "rolling_simd_block_table_avx512_scalars",
    "bytewise_simd_block_table_avx512_scalars",
    "rolling_simd_butterfly_table_neon_vectors",
    "bytewise_simd_butterfly_table_neon_vectors",
    "rolling_simd_butterfly_table_neon_scalars",
    "bytewise_simd_butterfly_table_neon_scalars",
    "rolling_simd_block_table_neon_vectors",
    "bytewise_simd_block_table_neon_vectors",
    "rolling_simd_block_table_neon_scalars",
    "bytewise_simd_block_table_neon_scalars",
    "rolling_simd_pext_vectors",
    "bytewise_simd_pext_vectors",
    "rolling_simd_pext_scalars",
    "bytewise_simd_pext_scalars",
    "rolling_simd_butterfly_table_scalar_vectors",
    "bytewise_simd_butterfly_table_scalar_vectors",
    "rolling_simd_butterfly_table_scalar_scalars",
    "bytewise_simd_butterfly_table_scalar_scalars",
    "rolling_simd_block_table_scalar_vectors",
    "bytewise_simd_block_table_scalar_vectors",
    "rolling_simd_block_table_scalar_scalars",
    "bytewise_simd_block_table_scalar_scalars",

    # DuoHash, with its naive as above
    "FSH",
    "ISSH",
    "FSH_multi",
    "MISSH_v1",
    "MISSH_col",
    "MISSH_col_parallel",
    "MISSH_row",

]

# The variants of the main manuscript, a subset of those of the supplement.
BENCHMARKS_MAIN = [
    # Bit extract implementations
    "naive",
    "pext",
    "rolling_scalar_pext",
    "bytewise_scalar_pext",
    "pext_scalar",
    # "bitloop",
    # "rolling_scalar_bitloop",
    # "bytewise_scalar_bitloop",
    # "split32",
    # "byte_table",
    # "rolling_scalar_byte_table",
    # "bytewise_scalar_byte_table",
    # "block_table",
    # "rolling_scalar_block_table",
    # "bytewise_scalar_block_table",
    # "block_table_unrolled2",
    # "rolling_scalar_block_table_unrolled2",
    # "bytewise_scalar_block_table_unrolled2",
    # "block_table_unrolled4",
    # "rolling_scalar_block_table_unrolled4",
    # "bytewise_scalar_block_table_unrolled4",
    # "block_table_unrolled8",
    # "rolling_scalar_block_table_unrolled8",
    # "bytewise_scalar_block_table_unrolled8",
    "butterfly_table",
    "rolling_scalar_butterfly_table",
    "bytewise_scalar_butterfly_table",
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
    "reextract_scalar_ascii",
    "rolling_scalar_ascii",
    "reextract_scalar_ifs",
    "rolling_scalar_ifs",
    "reextract_scalar_switch",
    "rolling_scalar_switch",
    "reextract_scalar_table",
    "rolling_scalar_table",
    # "pack_then_bytewise_scalar",
    "rolling_avx2",

    # Kmer spaced. Both forms: the vectors, and their single spaced k-mers, as the scalar
    # techniques hand them out.
    "rolling_simd_butterfly_table_sse2_vectors",
    "bytewise_simd_butterfly_table_sse2_vectors",
    "rolling_simd_butterfly_table_sse2_scalars",
    "bytewise_simd_butterfly_table_sse2_scalars",
    "rolling_simd_butterfly_table_avx2_vectors",
    "bytewise_simd_butterfly_table_avx2_vectors",
    "rolling_simd_butterfly_table_avx2_scalars",
    "bytewise_simd_butterfly_table_avx2_scalars",
    "rolling_simd_butterfly_table_avx512_vectors",
    "bytewise_simd_butterfly_table_avx512_vectors",
    "rolling_simd_butterfly_table_avx512_scalars",
    "bytewise_simd_butterfly_table_avx512_scalars",
    "rolling_simd_butterfly_table_neon_vectors",
    "bytewise_simd_butterfly_table_neon_vectors",
    "rolling_simd_butterfly_table_neon_scalars",
    "bytewise_simd_butterfly_table_neon_scalars",
]

# The labels of the variants whose name is not made of the parts of VARIANT_PARTS, such as
# benchmark-local and external code; more of them in the Sequence Packing section. All others are
# labeled by variant_label(), see benchmark_label().
BENCHMARK_RENAMES = {
    "naive"                     : "Naive",
    "missh"                     : "MISSH",
    "clark_original"            : "CLARK-S original",
    "clark_improved"            : "CLARK-S improved",
    "pack_then_bytewise_scalar" : "Pack, then bytewise scalar",
}

# The words of the parts of the variant names, per slot of the naming scheme, in the order of the
# slots in a name: algorithm, technique, encoder or extractor, internal helper, arch, and the form
# of the callback or other detail. A name uses any of the slots, each at most once.
VARIANT_PARTS = {
    "algorithm": {
        "rolling"   : "Rolling",
        "reextract" : "Reextract",
        "bytewise"  : "Bytewise",
        "chunked"   : "Chunked",
    },
    "technique": {
        "scalar" : "scalar",
        "simd"   : "SIMD",
        "swar"   : "SWAR",
    },
    "encoder": {
        # Character encoders
        "ascii"           : "ASCII",
        "ascii_checked"   : "ASCII checked",
        "ascii_unchecked" : "ASCII unchecked",
        "ifs"             : "'if'",
        "switch"          : "'switch'",
        "table"           : "Lookup table",
        # Word encoders
        "butterfly"     : "Butterfly",
        "direct"        : "Direct",
        "direct_double" : "DirectDouble",
        "fold"          : "Fold",
        # Bit extraction
        "pext"                  : "PEXT",
        "bitloop"               : "Bitloop",
        "split32"               : "Split32",
        "byte_table"            : "Byte Table",
        "block_table"           : "Block Table",
        "block_table_unrolled1" : "Block Table (1x unrolled)",
        "block_table_unrolled2" : "Block Table (2x unrolled)",
        "block_table_unrolled4" : "Block Table (4x unrolled)",
        "block_table_unrolled8" : "Block Table (8x unrolled)",
        "butterfly_table"       : "Butterfly Table",
        "instlatx"              : "InstLatX",
        "zp7"                   : "ZP7",
    },
    "helper": {
        "narrow_width" : "Narrow-Width",
        "full_width"   : "Full-Width",
    },
    "arch": {
        "scalar" : "scalar",
        "sse2"   : "SSE2",
        "avx2"   : "AVX2",
        "avx512" : "AVX512",
        "neon"   : "Neon",
    },
    "form": {
        "vectors"     : "vectors",
        "scalars"     : "scalars",
        "scratch"     : "scratch buffer",
        "threadlocal" : "thread-local buffer",
    },
}


def variant_label(name: str) -> str | None:
    """
    The label of a variant from the parts of its name, see VARIANT_PARTS, or None if the name is not
    made of them. The algorithm and technique come first, then, after a comma, the encoder or
    extractor, the helper and the arch, and the form in parentheses, such as "Rolling scalar, 'if'",
    "Bytewise SIMD Narrow-Width SSE2" or "Bytewise SIMD, Butterfly Table AVX2 (vectors)". Without an
    algorithm, the name is that of an encoder or extractor, such as "Direct SSE2".
    """
    words = {}
    rest = name
    for slot, parts in VARIANT_PARTS.items():
        # The longest part that the rest starts with, as parts can be prefixes of others.
        matches = [p for p in parts if rest == p or rest.startswith(p + "_")]
        if matches:
            part = max(matches, key=len)
            words[slot] = parts[part]
            rest = rest[len(part) + 1:]
    if rest or not words:
        return None

    head = " ".join(words[s] for s in ["algorithm", "technique"] if s in words)
    tail = " ".join(words[s] for s in ["encoder", "helper", "arch"] if s in words)
    sep = ", " if "algorithm" in words and "encoder" in words else " "
    label = sep.join(w for w in [head, tail] if w)
    if "form" in words:
        label += f" ({words['form']})"
    return label

# The short labels of the main mode, to keep the plots of the main manuscript simple.
BENCHMARK_RENAMES_MAIN = {
    # Kmer extract

    # Kmer spaced
    "naive"                           : "Naive",
    "pext"                            : "PEXT",
    "rolling_scalar_pext"             : "PEXT",
    "bytewise_scalar_pext"            : "PEXT",
    "pext_scalar"                     : "PEXT",
    "butterfly_table"                 : "Butterfly Table",
    "rolling_scalar_butterfly_table"  : "Butterfly Table",
    "bytewise_scalar_butterfly_table" : "Butterfly Table",

    "rolling_simd_butterfly_table_sse2_vectors"    : "SIMD SSE2 (vectors)",
    "bytewise_simd_butterfly_table_sse2_vectors"   : "SIMD SSE2 (vectors)",
    "rolling_simd_butterfly_table_sse2_scalars"    : "SIMD SSE2 (scalars)",
    "bytewise_simd_butterfly_table_sse2_scalars"   : "SIMD SSE2 (scalars)",
    "rolling_simd_butterfly_table_avx2_vectors"    : "SIMD AVX2 (vectors)",
    "bytewise_simd_butterfly_table_avx2_vectors"   : "SIMD AVX2 (vectors)",
    "rolling_simd_butterfly_table_avx2_scalars"    : "SIMD AVX2 (scalars)",
    "bytewise_simd_butterfly_table_avx2_scalars"   : "SIMD AVX2 (scalars)",
    "rolling_simd_butterfly_table_avx512_vectors"  : "SIMD AVX512 (vectors)",
    "bytewise_simd_butterfly_table_avx512_vectors" : "SIMD AVX512 (vectors)",
    "rolling_simd_butterfly_table_avx512_scalars"  : "SIMD AVX512 (scalars)",
    "bytewise_simd_butterfly_table_avx512_scalars" : "SIMD AVX512 (scalars)",
    "rolling_simd_butterfly_table_neon_vectors"    : "SIMD Neon (vectors)",
    "bytewise_simd_butterfly_table_neon_vectors"   : "SIMD Neon (vectors)",
    "rolling_simd_butterfly_table_neon_scalars"    : "SIMD Neon (scalars)",
    "bytewise_simd_butterfly_table_neon_scalars"   : "SIMD Neon (scalars)",

    # Kmer extract packed
    "bytewise_scalar"      : "Scalar",
    "bytewise_simd_sse2"   : "SIMD SSE2",
    "rolling_avx2"         : "Rolling AVX2",
    "bytewise_simd_avx2"   : "SIMD AVX2",
    "bytewise_simd_avx512" : "SIMD AVX512",
    "bytewise_simd_neon"   : "SIMD Neon",
}

# Stable colors for each implementation / benchmark
BENCHMARK_COLORS = {
    # Bit extract implementations
    "naive"                                 : "#23628f",
    "pext"                                  : "#5F5F5F",
    "rolling_scalar_pext"                   : "#5F5F5F",
    "bytewise_scalar_pext"                  : "#5F5F5F",
    "pext_scalar"                           : "#5F5F5F",
    "bitloop"                               : "#47a1e2",
    "rolling_scalar_bitloop"                : "#47a1e2",
    "bytewise_scalar_bitloop"               : "#47a1e2",
    "split32"                               : "#ff7f0e",
    "byte_table"                            : "#904ce9",
    "rolling_scalar_byte_table"             : "#904ce9",
    "bytewise_scalar_byte_table"            : "#904ce9",
    "block_table"                           : "#a1d99b",
    "rolling_scalar_block_table"            : "#a1d99b",
    "bytewise_scalar_block_table"           : "#a1d99b",
    "block_table_unrolled2"                 : "#74c476",
    "rolling_scalar_block_table_unrolled2"  : "#74c476",
    "bytewise_scalar_block_table_unrolled2" : "#74c476",
    "block_table_unrolled4"                 : "#31a354",
    "rolling_scalar_block_table_unrolled4"  : "#31a354",
    "bytewise_scalar_block_table_unrolled4" : "#31a354",
    "block_table_unrolled8"                 : "#006d2c",
    "rolling_scalar_block_table_unrolled8"  : "#006d2c",
    "bytewise_scalar_block_table_unrolled8" : "#006d2c",
    "butterfly_table"                       : "#BB2F98",
    "rolling_scalar_butterfly_table"        : "#BB2F98",
    "bytewise_scalar_butterfly_table"       : "#BB2F98",
    "instlatx"                              : "#000000",
    "zp7"                                   : "#000000",

    "reextract_scalar_ascii"    : "#3C5BBE",
    "rolling_scalar_ascii"      : "#3C5BBE",
    "ascii_checked"             : "#3C5BBE",
    "ascii_unchecked"           : "#3C5BBE",
    "reextract_scalar_ifs"      : "#E9C256",
    "rolling_scalar_ifs"        : "#E9C256",
    "reextract_scalar_switch"   : "#C53939",
    "rolling_scalar_switch"     : "#C53939",
    "reextract_scalar_table"    : "#6AC459",
    "rolling_scalar_table"      : "#6AC459",
    "pack_then_bytewise_scalar" : "#7F7F7F",
    "rolling_avx2"              : "#207542",
    "rolling_swar"              : "#9467BD",
    "chunked_scalar_threadlocal": "#000000",
    "chunked_scalar_scratch"    : "#000000",


    # Kmer spaced, from text and from packed sequences. scalars use a lighter tint of the vectors
    # color.
    "rolling_simd_butterfly_table_sse2_vectors"    : "#BB2F98",
    "bytewise_simd_butterfly_table_sse2_vectors"   : "#BB2F98",
    "rolling_simd_butterfly_table_sse2_scalars"    : "#E6A8D4",
    "bytewise_simd_butterfly_table_sse2_scalars"   : "#E6A8D4",
    "rolling_simd_block_table_sse2_vectors"        : "#006d2c",
    "bytewise_simd_block_table_sse2_vectors"       : "#006d2c",
    "rolling_simd_block_table_sse2_scalars"        : "#74c476",
    "bytewise_simd_block_table_sse2_scalars"       : "#74c476",
    "rolling_simd_butterfly_table_avx2_vectors"    : "#BB2F98",
    "bytewise_simd_butterfly_table_avx2_vectors"   : "#BB2F98",
    "rolling_simd_butterfly_table_avx2_scalars"    : "#E6A8D4",
    "bytewise_simd_butterfly_table_avx2_scalars"   : "#E6A8D4",
    "rolling_simd_block_table_avx2_vectors"        : "#006d2c",
    "bytewise_simd_block_table_avx2_vectors"       : "#006d2c",
    "rolling_simd_block_table_avx2_scalars"        : "#74c476",
    "bytewise_simd_block_table_avx2_scalars"       : "#74c476",
    "rolling_simd_butterfly_table_avx512_vectors"  : "#BB2F98",
    "bytewise_simd_butterfly_table_avx512_vectors" : "#BB2F98",
    "rolling_simd_butterfly_table_avx512_scalars"  : "#E6A8D4",
    "bytewise_simd_butterfly_table_avx512_scalars" : "#E6A8D4",
    "rolling_simd_block_table_avx512_vectors"      : "#006d2c",
    "bytewise_simd_block_table_avx512_vectors"     : "#006d2c",
    "rolling_simd_block_table_avx512_scalars"      : "#74c476",
    "bytewise_simd_block_table_avx512_scalars"     : "#74c476",
    "rolling_simd_butterfly_table_neon_vectors"    : "#BB2F98",
    "bytewise_simd_butterfly_table_neon_vectors"   : "#BB2F98",
    "rolling_simd_butterfly_table_neon_scalars"    : "#E6A8D4",
    "bytewise_simd_butterfly_table_neon_scalars"   : "#E6A8D4",
    "rolling_simd_block_table_neon_vectors"        : "#006d2c",
    "bytewise_simd_block_table_neon_vectors"       : "#006d2c",
    "rolling_simd_block_table_neon_scalars"        : "#74c476",
    "bytewise_simd_block_table_neon_scalars"       : "#74c476",
    "rolling_simd_pext_vectors"                    : "#5F5F5F",
    "bytewise_simd_pext_vectors"                   : "#5F5F5F",
    "rolling_simd_pext_scalars"                    : "#A8A8A8",
    "bytewise_simd_pext_scalars"                   : "#A8A8A8",
    "rolling_simd_butterfly_table_scalar_vectors"  : "#BB2F98",
    "bytewise_simd_butterfly_table_scalar_vectors" : "#BB2F98",
    "rolling_simd_butterfly_table_scalar_scalars"  : "#E6A8D4",
    "bytewise_simd_butterfly_table_scalar_scalars" : "#E6A8D4",
    "rolling_simd_block_table_scalar_vectors"      : "#006d2c",
    "bytewise_simd_block_table_scalar_vectors"     : "#006d2c",
    "rolling_simd_block_table_scalar_scalars"      : "#74c476",
    "bytewise_simd_block_table_scalar_scalars"     : "#74c476",
}

BENCHMARK_LINESTYLES = {
    # Extract / seq enc
    "reextract_scalar_ascii"    : "dashed",
    "rolling_scalar_ascii"      : "solid",
    "ascii_checked"             : "dashed",
    "ascii_unchecked"           : "solid",
    "reextract_scalar_ifs"      : "dashed",
    "rolling_scalar_ifs"        : "solid",
    "reextract_scalar_switch"   : "dashed",
    "rolling_scalar_switch"     : "solid",
    "reextract_scalar_table"    : "dashed",
    "rolling_scalar_table"      : "solid",
    "pack_then_bytewise_scalar" : "dashed",
    "chunked_scalar_threadlocal": "solid",
    "chunked_scalar_scratch"    : "dashed",
}

# Stable line order for plot consistency
BENCHMARK_ORDER = [

    # Extract / seq enc
    "reextract_scalar_ifs",
    "rolling_scalar_ifs",
    "reextract_scalar_switch",
    "rolling_scalar_switch",
    "reextract_scalar_ascii",
    "rolling_scalar_ascii",
    "reextract_scalar_table",
    "rolling_scalar_table",
    "pack_then_bytewise_scalar",
    "rolling_avx2",
    "rolling_swar",
    "chunked_scalar_threadlocal",
    "chunked_scalar_scratch",
    "ifs",
    "switch",
    "ascii_checked",
    "ascii_unchecked",
    "table",

    # Kmer spaced, also used for bit extract order
    "naive",
    "pext",
    "rolling_scalar_pext",
    "bytewise_scalar_pext",
    "pext_scalar",
    "bitloop",
    "rolling_scalar_bitloop",
    "bytewise_scalar_bitloop",
    # "split32",
    "byte_table",
    "rolling_scalar_byte_table",
    "bytewise_scalar_byte_table",
    "butterfly_table",
    "rolling_scalar_butterfly_table",
    "bytewise_scalar_butterfly_table",
    "block_table",
    "rolling_scalar_block_table",
    "bytewise_scalar_block_table",
    "block_table_unrolled1",
    "block_table_unrolled2",
    "rolling_scalar_block_table_unrolled2",
    "bytewise_scalar_block_table_unrolled2",
    "block_table_unrolled4",
    "rolling_scalar_block_table_unrolled4",
    "bytewise_scalar_block_table_unrolled4",
    "block_table_unrolled8",
    "rolling_scalar_block_table_unrolled8",
    "bytewise_scalar_block_table_unrolled8",
    # "instlatx",
    # "zp7",

    # Kmer spaced, from text and from packed sequences
    "rolling_simd_pext_vectors",
    "bytewise_simd_pext_vectors",
    "rolling_simd_pext_scalars",
    "bytewise_simd_pext_scalars",
    "rolling_simd_butterfly_table_scalar_vectors",
    "bytewise_simd_butterfly_table_scalar_vectors",
    "rolling_simd_butterfly_table_scalar_scalars",
    "bytewise_simd_butterfly_table_scalar_scalars",
    "rolling_simd_block_table_scalar_vectors",
    "bytewise_simd_block_table_scalar_vectors",
    "rolling_simd_block_table_scalar_scalars",
    "bytewise_simd_block_table_scalar_scalars",
    "rolling_simd_butterfly_table_sse2_vectors",
    "bytewise_simd_butterfly_table_sse2_vectors",
    "rolling_simd_butterfly_table_sse2_scalars",
    "bytewise_simd_butterfly_table_sse2_scalars",
    "rolling_simd_block_table_sse2_vectors",
    "bytewise_simd_block_table_sse2_vectors",
    "rolling_simd_block_table_sse2_scalars",
    "bytewise_simd_block_table_sse2_scalars",
    "rolling_simd_butterfly_table_avx2_vectors",
    "bytewise_simd_butterfly_table_avx2_vectors",
    "rolling_simd_butterfly_table_avx2_scalars",
    "bytewise_simd_butterfly_table_avx2_scalars",
    "rolling_simd_block_table_avx2_vectors",
    "bytewise_simd_block_table_avx2_vectors",
    "rolling_simd_block_table_avx2_scalars",
    "bytewise_simd_block_table_avx2_scalars",
    "rolling_simd_butterfly_table_avx512_vectors",
    "bytewise_simd_butterfly_table_avx512_vectors",
    "rolling_simd_butterfly_table_avx512_scalars",
    "bytewise_simd_butterfly_table_avx512_scalars",
    "rolling_simd_block_table_avx512_vectors",
    "bytewise_simd_block_table_avx512_vectors",
    "rolling_simd_block_table_avx512_scalars",
    "bytewise_simd_block_table_avx512_scalars",
    "rolling_simd_butterfly_table_neon_vectors",
    "bytewise_simd_butterfly_table_neon_vectors",
    "rolling_simd_butterfly_table_neon_scalars",
    "bytewise_simd_butterfly_table_neon_scalars",
    "rolling_simd_block_table_neon_vectors",
    "bytewise_simd_block_table_neon_vectors",
    "rolling_simd_block_table_neon_scalars",
    "bytewise_simd_block_table_neon_scalars",
]

# Stable order and colors for the kmer_extract_packed_* benchmark family (packed_*.hpp), kept as its
# own dict rather than folded into BENCHMARK_COLORS/BENCHMARK_ORDER above since it's a distinct
# naming scheme. Direct narrow-width/full-width rows are retained for code-generation comparisons;
# dispatchers are the user-facing entry points.
PACKED_KMER_VARIANT_ORDER = [
    "bytewise_scalar",
    "bytewise_simd_narrow_width_sse2",
    "bytewise_simd_full_width_sse2",
    "bytewise_simd_sse2",
    "bytewise_simd_narrow_width_avx2",
    "bytewise_simd_full_width_avx2",
    "bytewise_simd_avx2",
    "bytewise_simd_narrow_width_avx512",
    "bytewise_simd_full_width_avx512",
    "bytewise_simd_avx512",
    "bytewise_simd_narrow_width_neon",
    "bytewise_simd_full_width_neon",
    "bytewise_simd_neon",
    "bytewise_simd_narrow_width_scalar",
    "bytewise_simd_full_width_scalar",
    "bytewise_simd_scalar",
    "rolling_scalar",
]

PACKED_KMER_VARIANT_COLORS = {
    "bytewise_scalar" : "#107D84",
    "rolling_scalar" : "#636363",  # grey
    # SSE2: purple shades, ordered narrow-width -> full-width -> dispatcher.
    "bytewise_simd_narrow_width_sse2" : "#c5b0d5",
    "bytewise_simd_full_width_sse2" : "#9467bd",
    "bytewise_simd_sse2" : "#5C338A",
    # AVX2: green shades, ordered narrow-width -> full-width -> dispatcher.
    "bytewise_simd_narrow_width_avx2" : "#78c679",
    "bytewise_simd_full_width_avx2" : "#2ca02c",
    "bytewise_simd_avx2" : "#207542",
    # AVX-512: red shades, ordered narrow-width -> full-width -> dispatcher.
    "bytewise_simd_narrow_width_avx512" : "#fb6a4a",
    "bytewise_simd_full_width_avx512" : "#d62728",
    "bytewise_simd_avx512" : "#a23236",
    # Neon: blue shades, ordered narrow-width -> full-width -> dispatcher.
    "bytewise_simd_narrow_width_neon" : "#9ecae1",
    "bytewise_simd_full_width_neon" : "#6baed6",
    "bytewise_simd_neon" : "#23609d",
    # SimdScalar: brown shades, ordered narrow-width -> full-width -> dispatcher.
    "bytewise_simd_narrow_width_scalar" : "#dfc27d",
    "bytewise_simd_full_width_scalar" : "#bf812d",
    "bytewise_simd_scalar" : "#8c510a",
}

# Headline subset of PACKED_KMER_VARIANT_ORDER: the scalar baseline plus each ISA's public
# dispatcher only -- no narrow-width/full-width/rolling implementation detail. Each dispatcher
# already picks narrow-width or full-width internally depending on k, so a chart already split by
# k-tier (e.g. one figure per narrow-width/full-width range) loses nothing by showing only the
# dispatcher row: it *is* whichever of the two applies at that k. Used as the main view.
PACKED_KMER_VARIANT_ORDER_MAIN = [
    "bytewise_scalar",
    "bytewise_simd_sse2",
    "bytewise_simd_avx2",
    "bytewise_simd_avx512",
    "bytewise_simd_neon",
]

# Fold the packed k-mer variant names into the generic lists of the modes and BENCHMARK_ORDER too,
# so plot_bars_per_cpu.py's cross-CPU summary shows them instead of filtering every row out.
# Appended rather than merged in above since BENCHMARK_ORDER and the lists of the modes are defined
# earlier in this file, before PACKED_KMER_VARIANT_ORDER exists.
BENCHMARK_ORDER += PACKED_KMER_VARIANT_ORDER
BENCHMARKS_SUPP += PACKED_KMER_VARIANT_ORDER
BENCHMARKS_MAIN += PACKED_KMER_VARIANT_ORDER_MAIN


def add_mode_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --mode flag of every plotting script, see PLOT_MODES."""
    parser.add_argument(
        "--mode", choices=PLOT_MODES, default="full",
        help="Which variants to plot: the hand-picked ones of the main manuscript or of the "
             "supplement, or all of them (default: full).",
    )


def benchmarks_for_mode(mode: str, main=None, supp=None) -> set[str] | None:
    """The variants to plot in `mode`, or None for all of them (full). A script with a selection of
    its own passes it as `main` or `supp`; without one, those of BENCHMARKS_MAIN and
    BENCHMARKS_SUPP apply."""
    if mode == "full":
        return None
    if mode == "main":
        return set(BENCHMARKS_MAIN if main is None else main)
    if mode == "supp":
        return set(BENCHMARKS_SUPP if supp is None else supp)
    raise ValueError(f"Unknown mode {mode!r}, expected one of {PLOT_MODES}")


def filter_mode(df: pd.DataFrame, mode: str, main=None, supp=None) -> pd.DataFrame:
    """The rows of `df` with the variants of `mode`, see benchmarks_for_mode()."""
    keep = benchmarks_for_mode(mode, main, supp)
    return df if keep is None else df[df["benchmark"].isin(keep)]


def label_for_mode(name: str, mode: str) -> str:
    """The label of a variant: the short one of BENCHMARK_RENAMES_MAIN in the main mode, if any."""
    if mode == "main" and name in BENCHMARK_RENAMES_MAIN:
        return BENCHMARK_RENAMES_MAIN[name]
    return benchmark_label(name)


def benchmark_label(name: str) -> str:
    """The label of a variant: that of BENCHMARK_RENAMES, else variant_label(), else its name."""
    if name in BENCHMARK_RENAMES:
        return BENCHMARK_RENAMES[name]
    return variant_label(name) or name


# -------------------------------------------------------------------------------------------------
#     Sequence Packing
# -------------------------------------------------------------------------------------------------

# Benchmark name prefixes of the external reference implementations, see benchmarks/external.
EXTERNAL_PREFIXES = ("fastdna_", "cute_")

SEQUENCE_PACK_ORDER = [
    "fastdna_lut",
    "fastdna_pext_ordered",
    "fastdna_pext_unordered",
    "fastdna_pext_unordered_unroll2",
    "fastdna_pext_unordered_unroll4",
    "cute_mul",
    "cute_movemask",
    "pext_scalar", "pext_sse2", "pext_avx2", "pext_avx512",
    "butterfly_scalar", "butterfly_sse2", "butterfly_avx2", "butterfly_avx512", "butterfly_neon",
    "fold_scalar", "fold_sse2", "fold_avx2", "fold_avx512", "fold_neon",
    "direct_scalar", "direct_sse2", "direct_double_sse2",
    "direct_avx2", "direct_double_avx2", "direct_avx512", "direct_double_avx512",
    "direct_neon", "direct_double_neon",
]

# Explicit focused selection, easy to trim after collecting measurements.
SEQUENCE_PACK_FOCUSED = [
    "fastdna_lut",
    "fastdna_pext_ordered",
    "fastdna_pext_unordered",
    "fastdna_pext_unordered_unroll2",
    "fastdna_pext_unordered_unroll4",
    "cute_mul",
    "cute_movemask",
    "direct_scalar",
    "direct_sse2", "direct_double_sse2",
    "direct_avx2", "direct_double_avx2",
    "direct_avx512", "direct_double_avx512",
    "direct_neon", "direct_double_neon",
]

BENCHMARK_RENAMES.update({
    "fastdna_lut": "FastDNA\nLUT",
    "fastdna_pext_ordered": "FastDNA\nordered PEXT",
    "fastdna_pext_unordered": "FastDNA\nunordered PEXT",
    "fastdna_pext_unordered_unroll2": "FastDNA\nunordered PEXT (2x)",
    "fastdna_pext_unordered_unroll4": "FastDNA\nunordered PEXT (4x)",
    "cute_mul": "cute-nucleotides\nmul",
    "cute_movemask": "cute-nucleotides\nmovemask",
    "butterfly": "Butterfly scalar (legacy)",
})

SEQUENCE_PACK_COLORS = {
    "fastdna_lut": "#D9A066",
    "fastdna_pext_ordered": "#A65D23",
    "fastdna_pext_unordered": "#FFBB78",
    "fastdna_pext_unordered_unroll2": "#FF7F0E",
    "fastdna_pext_unordered_unroll4": "#B55B0B",
    "cute_mul": "#9E9AC8",
    "cute_movemask": "#6A51A3",
    "pext_scalar": BENCHMARK_COLORS["pext"],
    "pext_sse2": BENCHMARK_COLORS["pext"],
    "pext_avx2": BENCHMARK_COLORS["pext"],
    "pext_avx512": BENCHMARK_COLORS["pext"],
    "butterfly": "#BB2F98",
}


def _register_sequence_pack_colors():
    """One color per word encoder, for all of its archs."""
    for family, color in [("butterfly", "#BB2F98"), ("fold", "#74C476"), ("direct", "#3182BD")]:
        for arch in ["scalar", "sse2", "avx2", "avx512", "neon"]:
            SEQUENCE_PACK_COLORS[f"{family}_{arch}"] = color
    for arch in ["sse2", "avx2", "avx512", "neon"]:
        SEQUENCE_PACK_COLORS[f"direct_double_{arch}"] = "#08519C"


_register_sequence_pack_colors()
BENCHMARK_COLORS.update(SEQUENCE_PACK_COLORS)


# -----------------------------------------------------------------------------
#     Display unit: throughput (default) vs legacy ns/op
# -----------------------------------------------------------------------------

# Which "thing" one unit of ns_per_op actually measures, per benchmark suite (see each suite's
# .units_fn() in benchmarks/*/bench.cpp) -- used to build a correct throughput axis label. Suites
# not listed here (e.g. an old CSV from a removed suite, or a script that combines multiple
# suites) fall back to the generic "ops" noun in unit_noun_for_suite() below.
SUITE_UNIT_NOUN = {
    "kmer_extract_ascii_checked"                  : "k-mers",
    "kmer_extract_ascii_unchecked"                : "k-mers",
    "kmer_extract_packed_unchecked"               : "k-mers",
    "kmer_extract_packed_checked"                 : "k-mers",
    "kmer_clark"                                  : "k-mers",
    "spaced_kmer_extract_ascii_checked_single"    : "k-mers",
    "spaced_kmer_extract_ascii_checked_multi"     : "k-mers",
    "spaced_kmer_extract_packed_unchecked_single" : "k-mers",
    "spaced_kmer_extract_packed_unchecked_multi"  : "k-mers",
    "spaced_kmer_extract_packed_checked_single"   : "k-mers",
    "spaced_kmer_extract_packed_checked_multi"    : "k-mers",
    "char_encoder"                                : "bases",
    "sequence_pack_unchecked"                     : "bases",
    "sequence_pack_checked"                       : "bases",
    "bit_extract_weights"                         : "ops",
    "bit_extract_blocks"                          : "ops",
}

# The singular of each noun, for the time per unit: "Time per k-mer [ns]".
UNIT_NOUN_SINGULAR = {
    "k-mers" : "k-mer",
    "bases"  : "base",
    "ops"    : "operation",
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
    if suite in SUITE_UNIT_NOUN:
        return SUITE_UNIT_NOUN[suite]
    # The name of a pair of suites without the check, see single_suite().
    nouns = {n for s, n in SUITE_UNIT_NOUN.items() if without_check(s) == suite}
    return nouns.pop() if len(nouns) == 1 else "ops"


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


def single_suite(df, suite: str | None = None) -> str | None:
    """Return the suite of a plot, for its labels: `suite` if given, else the only suite in the
    rows of `df`, else None, as for plots of a single result file without --suite. The checked and
    unchecked suite of a pair, read together, count as one, under the name of both without the
    check, such as "kmer_extract_ascii"."""
    if suite is not None:
        return suite
    suites = {str(s) for s in df["suite"].dropna().unique()}
    if len(suites) == 1:
        return suites.pop()
    shared = {without_check(s) for s in suites}
    return shared.pop() if len(shared) == 1 else None


def without_check(suite: str) -> str:
    """The name of a suite without its check, which its checked and unchecked form share."""
    return re.sub(r"_(un)?checked(?=_|$)", "", suite)


def ylabel_for_unit(unit: str, suite: str | None = None) -> str:
    """Y-axis label for the selected --unit, both with the noun of the suite: "Time per <noun> [ns]"
    for "ns", such as "Time per k-mer [ns]", or "Throughput [G <noun>/s]" for "ops"."""
    noun = unit_noun_for_suite(suite)
    if unit == "ns":
        return f"Time per {UNIT_NOUN_SINGULAR.get(noun, noun)} [ns]"
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


# -------------------------------------------------------------------------------------------------
#     Axis grid plot
# -------------------------------------------------------------------------------------------------

def plot_axis_grid(
    csv_paths, load_pivot, axis_labels, axis_colors, default_title,
    outpath, unit, scale, y_min, y_max, title=None, xtick_label=lambda t: t,
):
    """
    Grid of grouped bar charts, one panel per CPU/compiler (one CSV each, or a list of the CSVs of
    one directory, which are read together): techniques on the x-axis, one bar per "axis" value
    within each technique, e.g. the vectors and single k-mers of a SIMD extraction, or several
    (encoding, layout) combinations.

    `load_pivot(csv_path, unit)` returns (suite, pivot), for one entry of `csv_paths`, with pivot
    index = technique, columns = axis values in display order, values in the display unit.
    `axis_labels`/`axis_colors` map each axis value to its legend label and bar color;
    `xtick_label` maps a technique to its tick label.
    """
    entries = [(p, platform_from_csv_path(p if isinstance(p, str) else p[0])) for p in csv_paths]
    entries.sort(key=lambda e: platform_compiler_sort_key(e[1]))

    loaded = []
    for csv_path, label in entries:
        suite, pivot = load_pivot(csv_path, unit)
        loaded.append((label, suite, pivot))

    suites = {suite for _, suite, _ in loaded}
    if len(suites) != 1:
        raise ValueError(f"All input files must be from the same suite, found {sorted(suites)}")
    suite = suites.pop()

    # One shared y-range across all panels, so they are directly comparable.
    all_values = np.concatenate([pivot.values.ravel() for _, _, pivot in loaded])
    axis_y_min, axis_y_max = compute_axis_limits(all_values, scale)
    # Extra top headroom for the rotated value labels above each bar, as in
    # plot_spaced_kmer_extract.py.
    axis_y_max *= 1.3 if scale == "log" else 1.1
    if y_min is not None:
        axis_y_min = y_min
    if y_max is not None:
        axis_y_max = y_max

    n = len(loaded)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    # At least 8 x 6 inches per panel; more for many bars, and for long, rotated technique labels,
    # such as those of all variants of the spaced suites.
    n_techniques = max(len(pivot.index) for _, _, pivot in loaded)
    n_bars = max(pivot.size for _, _, pivot in loaded)
    label_len = max(
        (len(xtick_label(t)) for _, _, pivot in loaded for t in pivot.index), default=0
    )
    panel_width = max(8, 0.35 * n_techniques + 0.12 * n_bars)
    panel_height = max(6, 3 + 0.1 * label_len)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(panel_width * ncols, 12), panel_height * nrows), squeeze=False,
        layout="constrained",
    )

    for idx, (label, _, pivot) in enumerate(loaded):
        ax = axes[idx // ncols][idx % ncols]
        apply_yscale(ax, scale)

        techniques = list(pivot.index)
        x = np.arange(len(techniques))
        axis_names = list(pivot.columns)
        bar_width = 0.8 / max(1, len(axis_names))
        offsets = [(j - (len(axis_names) - 1) / 2.0) * bar_width for j in range(len(axis_names))]

        for j, axis_name in enumerate(axis_names):
            vals = pivot[axis_name].values
            bars = ax.bar(
                x + offsets[j], np.nan_to_num(vals, nan=0.0), width=bar_width,
                label=axis_labels[axis_name], color=axis_colors[axis_name],
            )
            # Multiplicative headroom on "log", additive on "linear", as in
            # plot_spaced_kmer_extract.py.
            for bar, val in zip(bars, vals):
                if pd.notna(val):
                    label_y = (
                        val * 1.02 if scale == "log"
                        else val + (axis_y_max - axis_y_min) * 0.01
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, label_y, f"{val:.2f}",
                        ha="center", va="bottom", rotation=90, fontsize=9,
                    )

        ax.set_title(label)
        ax.set_ylabel(ylabel_for_unit(unit, suite))
        ax.set_xticks(x)
        ax.set_xticklabels([xtick_label(t) for t in techniques], rotation=45, ha="right")
        ax.set_ylim(axis_y_min, axis_y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide unused grid cells, if any.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # One shared legend below all panels, instead of one per panel, where it would cover bar labels.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title or f"{suite}: {default_title}")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))

    if outpath:
        fig.savefig(outpath, dpi=300)
        print(f"Wrote {outpath}")
    else:
        plt.show()
    plt.close(fig)
