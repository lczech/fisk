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
# Which --unit/--scale combinations to produce.
# Each entry is "unit:scale" -- comment lines out to skip a combination.
# Every plotting script suffixes its own --out filename with "_<unit>_<scale>"
# unconditionally, so different combos never clobber each other's output.
# --------------------------------------------------------------------

COMBOS=(
  "ops:log"       # default, giga-scale hero plots
  "ns:linear"     # legacy comparison
  # "ops:linear"
  # "ns:log"
)

# --------------------------------------------------------------------
# Plot calls
# --------------------------------------------------------------------

# Every plot invocation is independent (own CSV, own --out path, no shared
# state), so we run them concurrently, capped at MAX_JOBS, instead of one at
# a time -- run_plot backgrounds each call and "wait" at the very end blocks
# until they've all finished.
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# Run a plotting script for a given CSV, skipping gracefully (rather than
# crashing) if that benchmark hasn't been run for this CPU/compiler yet.
run_plot() {
  local csv="$1"; shift
  if [[ ! -f "$csv" ]]; then
    echo "Skipped ${csv} (not found)"
    return 0
  fi
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
  "$@" &
}

for EXT in png svg ; do
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"

    # Bit Extract Implementations

    CSV="${DIR}/bit_extract_weights.csv"
    run_plot "$CSV" python ./plot/plot_bit_extract_weights.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/bit_extract_weights.${EXT}"

    CSV="${DIR}/bit_extract_blocks.csv"
    run_plot "$CSV" python ./plot/plot_bit_extract_blocks.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/bit_extract_blocks.${EXT}"


    # Kmer Extract

    CSV="${DIR}/kmer_extract.csv"
    run_plot "$CSV" python ./plot/plot_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_extract.${EXT}"

    # python ./plot/plot_case_summary.py \
    #   "${DIR}/kmer_extract.csv" \
    #   --out "${DIR}/kmer_extract_bars.${EXT}"

    CSV="${DIR}/kmer_extract_packed.csv"
    # --y-max is unit-scale-dependent (throughput inverts by a different ratio per series, so a
    # ceiling tuned for ns/op can't just be reused for ops/s) -- 0.7 for ns/op, matching the
    # existing tuned value; 16 for ops, matching YLIM_PACKED_REDUCED in plot_all_cpus.sh (checked
    # against the real cross-CPU max throughput for this suite).
    PACKED_Y_MAX="16"
    [[ "$UNIT" == "ns" ]] && PACKED_Y_MAX="0.7"
    run_plot "$CSV" python ./plot/plot_kmer_extract_packed.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --y-max "$PACKED_Y_MAX" \
      --out "${DIR}/kmer_extract_packed.${EXT}"


    # Kmer Spaced Single

    CSV="${DIR}/kmer_spaced_single.csv"

    run_plot "$CSV" python ./plot/plot_kmer_spaced.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_single.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_single_bars.${EXT}"


    # Kmer Spaced Multi

    CSV="${DIR}/kmer_spaced_multi.csv"

    run_plot "$CSV" python ./plot/plot_kmer_spaced.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_multi.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_spaced_multi_bars.${EXT}"

  done
done

# Wait for every backgrounded plot job to finish before returning -- callers
# (e.g. plot_all_cpus.sh's per-directory loop, which converts the SVGs this
# script just wrote to PDF) rely on all output files existing once this
# script exits.
wait
