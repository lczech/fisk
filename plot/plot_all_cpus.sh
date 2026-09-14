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
ROOT="results"
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
# which is searched in the directories given above. Missing files are skipped so
# a benchmark collected on only some CPUs can still be plotted.
build_file_args() {
  local csv_name="$1"

  args=()
  for cpu in "${CPUS[@]}"; do
    local f="${ROOT}/${cpu}/${csv_name}"
    if [[ -f "$f" ]]; then
      args+=( --file "$f" )
    else
      echo "Skipping missing file: $f" >&2
    fi
  done
  if ((${#args[@]} == 0)); then
    echo "No ${csv_name} files found for the selected CPUs." >&2
    return 1
  fi
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
#     --file "results/AMD_Epyc_7763/kmer_extract.csv" \
#     --file "results/AMD_Ryzen_4750U/kmer_extract.csv" \
#     --file "results/Apple_M1/kmer_extract.csv" \
#     --file "results/Intel_Xeon_8568Y/kmer_extract.csv"

# ------------------------------------------------------------
# Kmer extract packed: detailed cross-platform comparison
# ------------------------------------------------------------

# Not in the generic CSV_FILES loop above: that loop applies the same (default-variant-set,
# unbounded y-axis) call to every CSV uniformly, but this suite needs its own y-axis limits and
# its regular (non-extended) call needs the reduced dispatcher-only variant set -- same reasoning
# as the "Manuscript figures" calls further below, which are for the same reason not in that loop
# either. Y-limits found by checking actual max ns_per_op across all current platforms.
YLIM_PACKED_REDUCED="0.75"
YLIM_PACKED_EXTENDED="1.2"

build_file_args "kmer_extract_packed.csv" || exit 1

# Regular: reduced variant set (scalar baseline + each ISA's dispatcher only), matching the
# per-CPU line plots' default and the narrow/wide tier plots below -- a summary, not full detail.
python ./plot/plot_bars_per_cpu.py "${args[@]}" \
  --reduced --y-lim "$YLIM_PACKED_REDUCED" \
  --out "${OUT}/kmer_extract_packed_per_cpu.png"

# Extended: every variant, including narrow/wide/rolling, for internal checking.
python ./plot/plot_bars_per_cpu.py "${args[@]}" \
  --extended --y-lim "$YLIM_PACKED_EXTENDED" \
  --out "${OUT}/kmer_extract_packed_per_cpu_ext.png"

# Complements the summary above (which averages over all cases) with per-(order,k) grouped bar
# charts and a platform summary, one variant per bar, so individual SIMD tiers/dispatchers stay
# distinguishable rather than collapsed into one mean.
# echo "Plotting kmer_extract_packed comparison"
# python ./plot/plot_kmer_extract_packed_compare.py "${args[@]}" \
#   --out-dir "$OUT"

# Same per-CPU grouped-bar style as the summary above (bars = platform/compiler, x-axis =
# implementation), but split by BitOrder and narrow/wide k-tier instead of blending everything into
# one mean. Reduced by default: at a fixed tier, each ISA's dispatcher bench already resolves to
# whichever of narrow/wide applies, so the narrow/wide/rolling detail rows would be redundant here.
for order in msb lsb; do
  python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    --case-filter "order=${order}" --case-filter "k<=29" \
    --reduced --y-lim "$YLIM_PACKED_REDUCED" \
    --out "${OUT}/kmer_extract_packed_per_cpu_${order}_narrow.png"

  python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    --case-filter "order=${order}" --case-filter "k>=30" --case-filter "k<=32" \
    --reduced --y-lim "$YLIM_PACKED_REDUCED" \
    --out "${OUT}/kmer_extract_packed_per_cpu_${order}_wide.png"
done

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

DUOHASH_SUITES=(
  "single"
  "multi"
)

for suite in "${DUOHASH_SUITES[@]}"; do

  # DuoHash
  csv="DuoHash.csv"
  build_file_args "$csv" || exit 1
  python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    --y-lim "25" \
    --y-lim "25" \
    --suite "${suite}" \
    --untight \
    --title "Spaced k-mer extraction, existing implementations, ${suite}" \
    --out "${OUT}/DuoHash-${suite}.svg"
  svg="${OUT}/DuoHash-${suite}.svg"
  inkscape "$svg" --export-filename="${svg%.svg}.pdf"

done

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
