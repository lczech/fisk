#!/usr/bin/env bash
set -euo pipefail

# Make all plots for a single CPU.
# Takes either the directory as input where our benchmark outputs are stored,
# or defaults to "benchmarks", which is where they are writting to by fisk.

# --------------------------------------------------------------------
# Directory argument (default: benchmarks)
# --------------------------------------------------------------------

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd `git rev-parse --show-toplevel`

# Silence some warnings
export QT_QPA_PLATFORM=xcb

if [[ $# -gt 1 ]]; then
  usage
fi

DIR="${1:-benchmarks}"
[[ -d "$DIR" ]] || { echo "Not a directory: $DIR" >&2; exit 1; }

DIR="${DIR%/}"
NAME="$(basename "$DIR")"

# OUTDIR="plots/${NAME}"
OUTDIR="${DIR}"
mkdir -p "$OUTDIR"

# --------------------------------------------------------------------
# Plot calls
# --------------------------------------------------------------------

for EXT in png svg ; do

  # Kmer Extract

  python ./plot/plot_kmer_extract.py \
    ${DIR}/kmer_extract.csv\
    --out ${DIR}/kmer_extract.${EXT}

  python ./plot/plot_impl_summary.py \
    ${DIR}/kmer_extract.csv\
    --out ${DIR}/kmer_extract_bars.${EXT}


  # Kmer Spaced

  python ./plot/plot_kmer_spaced.py \
    ${DIR}/kmer_spaced.csv\
    --out ${DIR}/kmer_spaced.${EXT}

  python ./plot/plot_impl_summary.py \
    ${DIR}/kmer_spaced.csv\
    --out ${DIR}/kmer_spaced_bars.${EXT}


  # PEXT Implementations

  python ./plot/plot_pext_blocks.py \
    ${DIR}/pext_blocks.csv\
    --out ${DIR}/pext_blocks.${EXT}

  python ./plot/plot_pext_weights.py \
    ${DIR}/pext_weights.csv\
    --out ${DIR}/pext_weights.${EXT}

done
