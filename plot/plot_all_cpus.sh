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
  # Standing in for the Apple entry until M1 Pro measurements exist -- swap for
  # "Apple M1 Pro, Clang 17" (currently has no kmer_extract_packed.csv) once available.
  "Apple M1, Clang 21"
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
# Which --unit/--scale combinations to produce.
# Each entry is "unit:scale" -- comment lines out to skip a combination.
# Every plotting script suffixes its own --out filename with "_<unit>_<scale>"
# unconditionally, so different combos never clobber each other's output.
# ------------------------------------------------------------

COMBOS=(
  "ops:log"       # default, giga-scale hero plots
  "ns:linear"     # legacy comparison
  # "ops:linear"
  # "ns:log"
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

# Every plot invocation is independent (own CSV, own --out path, no shared
# state), so we run them concurrently, capped at MAX_JOBS, instead of one at
# a time. run_job backgrounds a call; every block below that produces files
# some later step depends on (an inkscape SVG->PDF conversion, or the next
# section's own plots) is followed by a bare "wait" so those files are
# guaranteed to exist before anything reads them.
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

run_job() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
  "$@" &
}

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
  # "char_encoder.csv"
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
    for combo in "${COMBOS[@]}"; do
      UNIT="${combo%%:*}"
      SCALE="${combo##*:}"

      # Regular selection of CPUs and compilers
      run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
        --unit "$UNIT" --scale "$SCALE" \
        --out "${OUT}/${csv%.csv}_per_cpu.${EXT}"

      # Extended, with all available, for internal checking
      run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
        --extended \
        --unit "$UNIT" --scale "$SCALE" \
        --out "${OUT}/${csv%.csv}_per_cpu_ext.${EXT}"

      # Reduced set, mostly for the main manuscript
      # run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
      #   --reduced \
      #   --unit "$UNIT" --scale "$SCALE" \
      #   --out "${OUT}/${csv%.csv}_per_cpu_red.${EXT}"

    done
  done
  wait # every plot for this csv is on disk before converting svg -> pdf below

  # Convert to pdf if needed
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"

    svg="${OUT}/${csv%.csv}_per_cpu_${UNIT}_${SCALE}.svg"
    inkscape "$svg" --export-filename="${svg%.svg}.pdf"

    svg="${OUT}/${csv%.csv}_per_cpu_ext_${UNIT}_${SCALE}.svg"
    inkscape "$svg" --export-filename="${svg%.svg}.pdf"

    # svg="${OUT}/${csv%.csv}_per_cpu_red_${UNIT}_${SCALE}.svg"
    # inkscape "$svg" --export-filename="${svg%.svg}.pdf"
  done
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
# either. Y-limits found by checking actual max throughput/ns_per_op across all current platforms
# (--unit ops and --unit ns need separate constants: they're not a fixed ratio of each other since
# every series inverts by a different amount).
YLIM_PACKED_REDUCED_OPS="16"
YLIM_PACKED_EXTENDED_OPS="16"
YLIM_PACKED_REDUCED_NS="0.75"
YLIM_PACKED_EXTENDED_NS="1.2"

build_file_args "kmer_extract_packed.csv" || exit 1

for combo in "${COMBOS[@]}"; do
  UNIT="${combo%%:*}"
  SCALE="${combo##*:}"
  if [[ "$UNIT" == "ops" ]]; then
    YLIM_PACKED_REDUCED="$YLIM_PACKED_REDUCED_OPS"
    YLIM_PACKED_EXTENDED="$YLIM_PACKED_EXTENDED_OPS"
  else
    YLIM_PACKED_REDUCED="$YLIM_PACKED_REDUCED_NS"
    YLIM_PACKED_EXTENDED="$YLIM_PACKED_EXTENDED_NS"
  fi

  # Regular: reduced variant set (scalar baseline + each ISA's dispatcher only), matching the
  # per-CPU line plots' default and the narrow/wide tier plots below -- a summary, not full detail.
  run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    --reduced --unit "$UNIT" --scale "$SCALE" --y-max "$YLIM_PACKED_REDUCED" \
    --out "${OUT}/kmer_extract_packed_per_cpu.png"

  # Extended: every variant, including narrow/wide/rolling, for internal checking.
  run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
    --extended --unit "$UNIT" --scale "$SCALE" --y-max "$YLIM_PACKED_EXTENDED" \
    --out "${OUT}/kmer_extract_packed_per_cpu_ext.png"
done
wait

# Complements the summary above (which averages over all cases) with per-(layout,k) grouped bar
# charts and a platform summary, one variant per bar, so individual SIMD tiers/dispatchers stay
# distinguishable rather than collapsed into one mean.
# echo "Plotting kmer_extract_packed comparison"
# python ./plot/plot_kmer_extract_packed_compare.py "${args[@]}" \
#   --out-dir "$OUT"

# Same per-CPU grouped-bar style as the summary above (bars = platform/compiler, x-axis =
# implementation), but split by Layout and narrow/wide k-tier instead of blending everything into
# one mean. Reduced by default: at a fixed tier, each ISA's dispatcher bench already resolves to
# whichever of narrow/wide applies, so the narrow/wide/rolling detail rows would be redundant here.
for layout in msb lsb; do
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"
    if [[ "$UNIT" == "ops" ]]; then
      YLIM_PACKED_REDUCED="$YLIM_PACKED_REDUCED_OPS"
    else
      YLIM_PACKED_REDUCED="$YLIM_PACKED_REDUCED_NS"
    fi

    run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
      --case-filter "layout=${layout}" --case-filter "k<=29" \
      --reduced --unit "$UNIT" --scale "$SCALE" --y-max "$YLIM_PACKED_REDUCED" \
      --out "${OUT}/kmer_extract_packed_per_cpu_${layout}_narrow.png"

    run_job python ./plot/plot_bars_per_cpu.py "${args[@]}" \
      --case-filter "layout=${layout}" --case-filter "k>=30" --case-filter "k<=32" \
      --reduced --unit "$UNIT" --scale "$SCALE" --y-max "$YLIM_PACKED_REDUCED" \
      --out "${OUT}/kmer_extract_packed_per_cpu_${layout}_wide.png"
  done
done
wait

# ------------------------------------------------------------
# Manuscript figures
# ------------------------------------------------------------

# Same y-max for both figures, so they stay directly comparable -- see the note on
# YLIM_PACKED_REDUCED_OPS/_NS above for why --unit ops and --unit ns need separate constants.
Y_LIM_OPS="1.8"
Y_LIM_NS="6.0"

build_file_args "kmer_spaced_single.csv" || exit 1
args_single=("${args[@]}")
build_file_args "kmer_spaced_multi.csv" || exit 1
args_multi=("${args[@]}")

for combo in "${COMBOS[@]}"; do
  UNIT="${combo%%:*}"
  SCALE="${combo##*:}"
  Y_LIM="$Y_LIM_OPS"
  [[ "$UNIT" == "ns" ]] && Y_LIM="$Y_LIM_NS"

  # Single mask spaced kmers
  run_job python ./plot/plot_bars_per_cpu.py "${args_single[@]}" \
    --reduced \
    --unit "$UNIT" --scale "$SCALE" --y-max "$Y_LIM" \
    --no-legend \
    --title "(a) Extraction from sequence to spaced k-mers with a single mask" \
    --out "${OUT}/Fig2a.svg"

  # Multi mask spaced kmers
  run_job python ./plot/plot_bars_per_cpu.py "${args_multi[@]}" \
    --reduced \
    --unit "$UNIT" --scale "$SCALE" --y-max "$Y_LIM" \
    --title "(b) Extraction from sequence to spaced k-mers with multiple masks" \
    --out "${OUT}/Fig2b.svg"
done
wait

for combo in "${COMBOS[@]}"; do
  UNIT="${combo%%:*}"
  SCALE="${combo##*:}"

  svg="${OUT}/Fig2a_${UNIT}_${SCALE}.svg"
  inkscape "$svg" --export-filename="${svg%.svg}.pdf"

  svg="${OUT}/Fig2b_${UNIT}_${SCALE}.svg"
  inkscape "$svg" --export-filename="${svg%.svg}.pdf"
done

DUOHASH_YLIM_OPS="0.3"
DUOHASH_YLIM_NS="25"
DUOHASH_SUITES=(
  "single"
  "multi"
)

build_file_args "DuoHash.csv" || exit 1
args_duohash=("${args[@]}")

for suite in "${DUOHASH_SUITES[@]}"; do
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"
    DUOHASH_YLIM="$DUOHASH_YLIM_OPS"
    [[ "$UNIT" == "ns" ]] && DUOHASH_YLIM="$DUOHASH_YLIM_NS"

    # DuoHash
    run_job python ./plot/plot_bars_per_cpu.py "${args_duohash[@]}" \
      --unit "$UNIT" --scale "$SCALE" --y-max "$DUOHASH_YLIM" \
      --suite "${suite}" \
      --untight \
      --title "Spaced k-mer extraction, existing implementations, ${suite}" \
      --out "${OUT}/DuoHash-${suite}.svg"
  done
done
wait

for suite in "${DUOHASH_SUITES[@]}"; do
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"
    svg="${OUT}/DuoHash-${suite}_${UNIT}_${SCALE}.svg"
    inkscape "$svg" --export-filename="${svg%.svg}.pdf"
  done
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
  #
  # Only the SVGs plot_all_benchs.sh's current COMBOS just (re)wrote -- matched by their
  # "_<unit>_<scale>" suffix -- not a blind "*.svg" glob: a directory can still hold older SVGs
  # from before this suffix existed (or from a combo since commented out), and those are never
  # regenerated by the pipeline anymore, so reconverting them here would just re-encode an
  # unrelated, unchanged file (inkscape's output isn't byte-stable across runs) and produce a
  # spurious diff on a tracked PDF that this run never actually touched.
  echo "Converting SVG to PDF"
  for combo in "${COMBOS[@]}"; do
    UNIT="${combo%%:*}"
    SCALE="${combo##*:}"
    for svg in "${ROOT}/${cpu}"/*"_${UNIT}_${SCALE}.svg"; do
      [[ -e "$svg" ]] || continue
      inkscape "$svg" --export-filename="${svg%.svg}.pdf"
    done
  done
done
