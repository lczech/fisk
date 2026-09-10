#!/usr/bin/env bash
set -euo pipefail

# Make all plots for a single CPU.
# Takes either the directory as input where our benchmark outputs are stored,
# or defaults to "results", which is where they are writting to by fisk.

# --------------------------------------------------------------------
# Directory argument (default: results)
# --------------------------------------------------------------------

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd `git rev-parse --show-toplevel`

# Silence some warnings
export QT_QPA_PLATFORM=xcb

if [[ $# -gt 1 ]]; then
  usage
fi

DIR="${1:-results}"
[[ -d "$DIR" ]] || { echo "Not a directory: $DIR" >&2; exit 1; }

DIR="${DIR%/}"
NAME="$(basename "$DIR")"

# OUTDIR="plots/${NAME}"
OUTDIR="${DIR}"
mkdir -p "$OUTDIR"

# --------------------------------------------------------------------
# Plot calls
# --------------------------------------------------------------------

# Run a plotting script for a given CSV, skipping gracefully (rather than
# crashing) if that benchmark hasn't been run for this CPU/compiler yet.
run_plot() {
  local csv="$1"; shift
  if [[ ! -f "$csv" ]]; then
    echo "Skipped ${csv} (not found)"
    return 0
  fi
  "$@"
}

for EXT in png svg ; do

  # Bit Extract Implementations

  CSV="${DIR}/bit_extract_weights.csv"
  run_plot "$CSV" python ./plot/plot_bit_extract_weights.py \
    "$CSV" \
    --out "${DIR}/bit_extract_weights.${EXT}"

  CSV="${DIR}/bit_extract_blocks.csv"
  run_plot "$CSV" python ./plot/plot_bit_extract_blocks.py \
    "$CSV" \
    --out "${DIR}/bit_extract_blocks.${EXT}"


  # Kmer Extract

  CSV="${DIR}/kmer_extract.csv"
  run_plot "$CSV" python ./plot/plot_kmer_extract.py \
    "$CSV" \
    --out "${DIR}/kmer_extract.${EXT}"

  # python ./plot/plot_case_summary.py \
  #   "${DIR}/kmer_extract.csv" \
  #   --out "${DIR}/kmer_extract_bars.${EXT}"


  # Kmer Spaced Single

  CSV="${DIR}/kmer_spaced_single.csv"

  run_plot "$CSV" python ./plot/plot_kmer_spaced.py \
    "$CSV" \
    --out "${DIR}/kmer_spaced_single.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --out "${DIR}/kmer_spaced_single_bars.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --extended \
    --out "${DIR}/kmer_spaced_single_bars.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --reduced \
    --out "${DIR}/kmer_spaced_single_bars.${EXT}"


  # Kmer Spaced Multi

  CSV="${DIR}/kmer_spaced_multi.csv"

  run_plot "$CSV" python ./plot/plot_kmer_spaced.py \
    "$CSV" \
    --out "${DIR}/kmer_spaced_multi.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --extended \
    --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

  run_plot "$CSV" python ./plot/plot_case_summary.py \
    "$CSV" \
    --reduced \
    --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

done
