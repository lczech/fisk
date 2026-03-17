#!/usr/bin/env bash
set -euo pipefail

# We have directories with files with identical names in them.
# This script collets those files for a given input name,
# and copies them into an output dir of the file name,
# while nameing the files after their output directory instead.
# Thus, `AMD-EPYC/k-mer-extract.png` becomes `k-mer-extract/AMD-EPYC.png`,
# which helps us collect all plots for the same benchmark,
# for ease of analysis.

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd `git rev-parse --show-toplevel`

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

fname="$1"
base="${fname%.*}"          # filename without extension
ext="${fname##*.}"          # extension
indir="benchmarks"
outdir="${indir}/Collect/${base}"

mkdir -p "$outdir"

# find all occurrences of the file in subdirectories
find "$indir" -type f -name "$fname" -print0 | while IFS= read -r -d '' file; do
    dir="$(basename "$(dirname "$file")")"
    cp "$file" "$outdir/$dir.$ext"
done
