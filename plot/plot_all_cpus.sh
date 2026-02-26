#!/usr/bin/env bash
set -euo pipefail

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd `git rev-parse --show-toplevel`

# Silence some warnings
export QT_QPA_PLATFORM=xcb

# ------------------------------------------------------------
# Define explicit directories.
# Leave empty to auto-scan.
# ------------------------------------------------------------

ROOT="benchmarks"
CPUS=(
  "AMD_Epyc_7763"
  "AMD_Ryzen_4750U"
  "Apple_M1"
  "Intel_Xeon_8568Y"
)

if ((${#CPUS[@]} == 0)); then
  mapfile -t CPUS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi
if ((${#CPUS[@]} == 0)); then
  echo "No benchmark directories found." >&2
  exit 1
fi

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

# Build the arguments --file A --file B for a given CSV name,
# which is searched in the directories given above.
build_file_args() {
  local csv_name="$1"

  args=()
  for cpu in "${CPUS[@]}"; do
    local f="${ROOT}/${cpu}/${csv_name}"
    [[ -f "$f" ]] || { echo "Missing file: $f" >&2; return 1; }
    args+=( --file "$f" )
  done
}

# ------------------------------------------------------------
# Combined call (all CPUs together)
# ------------------------------------------------------------

# Benchmarks for which we want the summary plot across CPUs.
CSV_FILES=(
  "seq_enc.csv"
  "kmer_extract.csv"
)

for csv in "${CSV_FILES[@]}"; do
  build_file_args "$csv" || exit 1

  for EXT in png svg ; do

    python ./plot/plot_bars_per_cpu.py "${args[@]}" \
        --out "${ROOT}/${csv%.csv}_per_cpu.${EXT}"

  done

done

# Simple example for reference of what we are doing
# ./plot/plot_bars_per_cpu.py \
#     --file "benchmarks/AMD_Epyc_7763/kmer_extract.csv" \
#     --file "benchmarks/AMD_Ryzen_4750U/kmer_extract.csv" \
#     --file "benchmarks/Apple_M1/kmer_extract.csv" \
#     --file "benchmarks/Intel_Xeon_8568Y/kmer_extract.csv"

# ------------------------------------------------------------
# Per-directory loop
# ------------------------------------------------------------

# Run the plotting scripts in all CPU directories.
# This is for the indivdual plots per CPU.
# for cpu in "${CPUS[@]}"; do
#   ./plot/plot_all_benchs.sh "${ROOT}/${cpu}"
# done
