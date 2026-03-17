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

# Select which CPUs we want in the summary plots
ROOT="benchmarks"
CPUS=(
  # "AMD EPYC 7763, Clang 18"
  # "AMD EPYC 7763, GCC 13"
  "AMD EPYC 9684X, Clang 17"
  "AMD EPYC 9684X, GCC 15"
  "AMD Ryzen 7 Pro 4750U, Clang 17"
  "AMD Ryzen 7 Pro 4750U, GCC 14"
  "Intel Xeon Platinum 8568Y, Clang 17"
  "Intel Xeon Platinum 8568Y, GCC 15"
  "Apple M1 Pro, Clang 17"
)

if ((${#CPUS[@]} == 0)); then
  mapfile -t CPUS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi
if ((${#CPUS[@]} == 0)); then
  echo "No benchmark directories found." >&2
  exit 1
fi

OUT="${ROOT}/Summaries"
mkdir -p "$OUT"

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
  "bit_extract_weights.csv"
  "bit_extract_blocks.csv"
  # "seq_enc.csv"
  "kmer_extract.csv"
  "kmer_spaced_multi.csv"
  "kmer_spaced_single.csv"
  # "kmer_clark.csv"
)

# Which output formats do we want to produce?
# For now, just png, which svg optional if needed to refine figures later.
FORMATS=(
  "png"
  "svg"
)

echo "Plotting summaries"
for csv in "${CSV_FILES[@]}"; do
  build_file_args "$csv" || exit 1

  for EXT in "${FORMATS[@]}"; do

    # Regular selection of CPUs and compilers
    python ./plot/plot_bars_per_cpu.py "${args[@]}" \
      --out "${OUT}/${csv%.csv}_per_cpu.${EXT}"

    # Extended, with all available, for internal checking
    python ./plot/plot_bars_per_cpu.py "${args[@]}" \
      --extended \
      --out "${OUT}/${csv%.csv}_per_cpu_ext.${EXT}"

    # Reduced set, mostly for the main manuscript
    # python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    #   --reduced \
    #   --out "${OUT}/${csv%.csv}_per_cpu_red.${EXT}"

  done

  # Convert ot pdf if needed

  svg="${OUT}/${csv%.csv}_per_cpu.svg"
  inkscape "$svg" --export-filename="${svg%.svg}.pdf"

  svg="${OUT}/${csv%.csv}_per_cpu_ext.svg"
  inkscape "$svg" --export-filename="${svg%.svg}.pdf"

  # svg="${OUT}/${csv%.csv}_per_cpu_red.svg"
  # inkscape "$svg" --export-filename="${svg%.svg}.pdf"

done

# Simple example for reference of what we are doing
# ./plot/plot_bars_per_cpu.py \
#     --file "benchmarks/AMD_Epyc_7763/kmer_extract.csv" \
#     --file "benchmarks/AMD_Ryzen_4750U/kmer_extract.csv" \
#     --file "benchmarks/Apple_M1/kmer_extract.csv" \
#     --file "benchmarks/Intel_Xeon_8568Y/kmer_extract.csv"

# ------------------------------------------------------------
# Manuscript figures
# ------------------------------------------------------------

# Same y lim for both figures
Y_LIM="6.0"

# Single mask spaced kmers
csv="kmer_spaced_single.csv"
build_file_args "$csv" || exit 1
python ./plot/plot_bars_per_cpu.py "${args[@]}" \
  --reduced \
  --y-lim "$Y_LIM" \
  --no-legend \
  --title "(a) Extraction from sequence to spaced k-mers with a single mask" \
  --out "${OUT}/Fig2a.svg"
svg="${OUT}/Fig2a.svg"
inkscape "$svg" --export-filename="${svg%.svg}.pdf"


# Multi mask spaced kmers
csv="kmer_spaced_multi.csv"
build_file_args "$csv" || exit 1
python ./plot/plot_bars_per_cpu.py "${args[@]}" \
  --reduced \
  --y-lim "$Y_LIM" \
  --title "(b) Extraction from sequence to spaced k-mers with multiple masks" \
  --out "${OUT}/Fig2b.svg"
svg="${OUT}/Fig2b.svg"
inkscape "$svg" --export-filename="${svg%.svg}.pdf"

# ------------------------------------------------------------
# Per-directory loop
# ------------------------------------------------------------

# Don't plot all individual directories again.
# exit 0

# Get ALL sub-directories of root that contain a `sys_info.txt` file.
mapfile -t CPUS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/sys_info.txt" \; -printf "%f\n" | sort)

# Run the plotting scripts in all CPU directories.
# This is for the indivdual plots per CPU.
for cpu in "${CPUS[@]}"; do
  echo
  echo "Plotting ${cpu}"
  ./plot/plot_all_benchs.sh "${ROOT}/${cpu}"

  # This will generate individual plots for each benchmark in the CPU directory.
  # Next, we use inkscape to convert all of them to PDF.
  echo "Converting SVG to PDF"
  for svg in "${ROOT}/${cpu}"/*.svg; do
    inkscape "$svg" --export-filename="${svg%.svg}.pdf"
  done
done
