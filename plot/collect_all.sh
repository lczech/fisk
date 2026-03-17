#!/usr/bin/env bash
set -euo pipefail

# We have directories with files with identical names in them.
# This script collets those files for a given input name,
# and pivots the name, such that it is easier to see plots
# for the same benchmark across different CPUs.

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd `git rev-parse --show-toplevel`

# List of benchmarks we want to collect
BENCHMARKS=(
    "bit_extract_weights"
    "bit_extract_blocks"
    # "seq_enc"
    "kmer_extract"
    "kmer_spaced_multi_bars"
    "kmer_spaced_single_bars"
    # "kmer_clark"
)

# List of extensions whose files we want
EXTENSIONS=(
    "png"
    "pdf"
)

# Run the collect script for all combinations
for benchmark in "${BENCHMARKS[@]}"; do
    for ext in "${EXTENSIONS[@]}"; do
        ./plot/collect.sh "$benchmark.$ext"
    done
done
