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
source plot/log.sh

# Silence some warnings
export QT_QPA_PLATFORM=xcb

usage() {
  echo "Usage: $0 [results_dir]  (default: results)" >&2
  exit 1
}

if [[ $# -gt 1 ]]; then
  usage
fi

DIR="${1:-results}"
[[ -d "$DIR" ]] || { error "Not a directory: $DIR"; exit 1; }

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
# nproc is GNU-only (missing on macOS); getconf covers macOS/BSD.
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

# Block until a job slot frees up. "wait -n" needs bash >= 4.3; the stock
# macOS bash (3.2) lacks it, so there we drain the whole batch instead.
wait_any() {
  if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
    wait -n
  else
    wait
  fi
}

# Run a plotting script for a given CSV, skipping gracefully (rather than
# crashing) if that benchmark hasn't been run for this CPU/compiler yet.
run_plot() {
  local csv="$1"; shift
  if [[ ! -f "$csv" ]]; then
    warn "${csv} missing, skipping its plot"
    return 0
  fi
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait_any
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


    # Sequence Packing, without and with validity

    for SUITE in sequence_pack_unchecked sequence_pack_checked; do
      CSV="${DIR}/${SUITE}.csv"
      run_plot "$CSV" python ./plot/plot_sequence_pack.py \
        "$CSV" --unit "$UNIT" --scale "$SCALE" --format "$EXT" \
        --out-dir "$DIR"
    done


    # Kmer Extract, from ASCII: the checked and the unchecked suite in one plot

    CSV="${DIR}/kmer_extract_ascii_checked.csv"
    run_plot "$CSV" python ./plot/plot_kmer_extract_ascii.py \
      "$CSV" "${DIR}/kmer_extract_ascii_unchecked.csv" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_extract_ascii.${EXT}"

    # python ./plot/plot_case_summary.py \
    #   "${DIR}/kmer_extract_ascii_checked.csv" \
    #   --out "${DIR}/kmer_extract_ascii_checked_bars.${EXT}"

    CSV="${DIR}/kmer_extract_packed_unchecked.csv"
    # --y-max is unit-scale-dependent (throughput inverts by a different ratio per series, so a
    # ceiling tuned for ns/op can't just be reused for ops/s) -- 0.7 for ns/op, matching the
    # existing tuned value; 16 for ops, matching YLIM_PACKED_REDUCED in plot_all_cpus.sh (checked
    # against the real cross-CPU max throughput for this suite).
    # The checked suite uses the same limit, so that the two stay comparable.
    PACKED_Y_MAX="16"
    [[ "$UNIT" == "ns" ]] && PACKED_Y_MAX="0.7"
    run_plot "$CSV" python ./plot/plot_kmer_extract_packed.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --y-max "$PACKED_Y_MAX" \
      --out "${DIR}/kmer_extract_packed_unchecked.${EXT}"

    CSV="${DIR}/kmer_extract_packed_checked.csv"
    run_plot "$CSV" python ./plot/plot_kmer_extract_packed.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --y-max "$PACKED_Y_MAX" \
      --out "${DIR}/kmer_extract_packed_checked.${EXT}"

    # The vectors and single k-mers of the checked SIMD extraction. The cross-CPU plots of both
    # suites are in plot_all_cpus.sh.
    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/kmer_extract_packed_checked_vectors_scalars.${EXT}"


    # Kmer Spaced Single

    CSV="${DIR}/spaced_kmer_extract_ascii_checked_single.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_single.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_single_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_single_bars.${EXT}"


    # Kmer Spaced Multi

    CSV="${DIR}/spaced_kmer_extract_ascii_checked_multi.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_multi.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_multi_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_ascii_checked_multi_bars.${EXT}"


    # Kmer Spaced Packed Single

    CSV="${DIR}/spaced_kmer_extract_packed_unchecked_single.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_single.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_single_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_single_bars.${EXT}"


    # Kmer Spaced Packed Multi

    CSV="${DIR}/spaced_kmer_extract_packed_unchecked_multi.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_multi.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_multi_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_unchecked_multi_bars.${EXT}"

    # Kmer Spaced Packed Checked Single

    CSV="${DIR}/spaced_kmer_extract_packed_checked_single.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_single.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_single_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_single_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_single_bars.${EXT}"


    # Kmer Spaced Packed Checked Multi

    CSV="${DIR}/spaced_kmer_extract_packed_checked_multi.csv"

    run_plot "$CSV" python ./plot/plot_spaced_kmer_extract.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_multi.${EXT}"

    run_plot "$CSV" python ./plot/plot_vectors_scalars_per_cpu.py \
      --file "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_multi_vectors_scalars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --extended \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_multi_bars.${EXT}"

    run_plot "$CSV" python ./plot/plot_case_summary.py \
      "$CSV" \
      --reduced \
      --unit "$UNIT" --scale "$SCALE" \
      --out "${DIR}/spaced_kmer_extract_packed_checked_multi_bars.${EXT}"


    # Checked vs. Unchecked

    # The unchecked suites that have a checked counterpart with the same techniques, as
    # "<name> <unchecked csv>", where <name> is the name of both without the check, and the checked
    # csv is the unchecked one with "_checked" for "_unchecked". The plot is skipped if one of the
    # two tables is missing.
    for pair in \
      "spaced_kmer_extract_packed_single spaced_kmer_extract_packed_unchecked_single.csv" \
      "spaced_kmer_extract_packed_multi spaced_kmer_extract_packed_unchecked_multi.csv" \
      "kmer_extract_packed kmer_extract_packed_unchecked.csv" \
      "sequence_pack sequence_pack_unchecked.csv"
    do
      read -r name unchecked <<< "$pair"
      checked="${unchecked/_unchecked/_checked}"
      run_plot "${DIR}/${unchecked}" python ./plot/plot_checked_vs_unchecked_per_cpu.py \
        --dir "$DIR" --unchecked "$unchecked" --checked "$checked" \
        --unit "$UNIT" --scale "$SCALE" \
        --out "${DIR}/${name}_checked_vs_unchecked.${EXT}"
    done

  done
done

# Wait for every backgrounded plot job to finish before returning -- callers
# (e.g. plot_all_cpus.sh's per-directory loop, which converts the SVGs this
# script just wrote to PDF) rely on all output files existing once this
# script exits.
wait
