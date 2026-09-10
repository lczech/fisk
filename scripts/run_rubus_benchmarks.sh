#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
#   Fisk benchmark runner for the rubus cluster
# ==============================================================================
#
# Runs the fisk benchmarks across all combinations of CPU arch x compiler:
#   - AMD EPYC 9684X    x  Clang 17 / GCC 15
#   - Intel Xeon 8568Y  x  Clang 17 / GCC 15
#
# Usage:
#   Run from a working directory that already contains a `fisk/` subdirectory
#   with exactly the source tree to build - either a fresh `git clone`, or an
#   unzipped dev copy built from `git ls-files`.
#
#     ./run_rubus_benchmarks.sh
#
# On completion, results land in ./results/<CPU>, <Compiler>/ (matching the
# fisk repo existing results/ naming convention), logs in ./logs/. Rerunning
# archives any previous results/logs for a combo under a timestamped
# subdirectory instead of overwriting them.

# ------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------

ACCOUNT="spear"
QOS="normal"
TIME_LIMIT="2:00:00"

# Node constraints, per CPU arch. "std" excludes the gpu/hgx nodes.
declare -A CONSTRAINT=(
  [epyc]="std,epyc,znver4"
  [xeon]="std,xeon,sapphirerapids"
)

# Module to load, per compiler.
declare -A MODULE=(
  [clang]="Clang"
  [gcc]="GCC/15.2.0"
)

# CC / CXX to export, per compiler.
declare -A CC_BIN=(  [clang]="clang"   [gcc]="gcc" )
declare -A CXX_BIN=( [clang]="clang++" [gcc]="g++" )

# Human-readable labels, per arch/compiler - combined below into the same
# "<CPU>, <Compiler>" directory names already used under the fisk repo's
# results/, so plot_all_cpus.sh picks these up with no renaming needed.
declare -A CPU_LABEL=(
  [epyc]="AMD EPYC 9684X"
  [xeon]="Intel Xeon Platinum 8568Y"
)
declare -A COMPILER_LABEL=(
  [clang]="Clang 17"
  [gcc]="GCC 15"
)

# The four combos to run. Each gets its own --exclusive node, so all four
# run fully in parallel with no timing interference between them.
COMBOS=(
  "epyc:clang"
  "epyc:gcc"
  "xeon:clang"
  "xeon:gcc"
)

# ------------------------------------------------------------------------
#   Setup
# ------------------------------------------------------------------------

if [[ ! -d "fisk" ]]; then
  echo "Error: no 'fisk' directory found in $(pwd)." >&2
  echo "Run this from the working directory that contains it." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p results logs

STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# ------------------------------------------------------------------------
#   One combo: fresh copy, build, run, archive-then-write results/logs.
# ------------------------------------------------------------------------

run_combo() {
  local arch="$1" compiler="$2"
  local slug="${arch}-${compiler}"
  local label="${CPU_LABEL[$arch]}, ${COMPILER_LABEL[$compiler]}"
  local constraint="${CONSTRAINT[$arch]}"
  local module="${MODULE[$compiler]}"
  local cc="${CC_BIN[$compiler]}"
  local cxx="${CXX_BIN[$compiler]}"
  local workdir="fisk-${slug}"
  local resultdir="results/${label}"
  local logfile="logs/${slug}.log"

  # Archive any previous results/log for this combo instead of overwriting.
  if [[ -d "$resultdir" ]]; then
    mkdir -p "results/_archive/${TIMESTAMP}"
    mv "$resultdir" "results/_archive/${TIMESTAMP}/${label}"
  fi
  if [[ -f "$logfile" ]]; then
    mkdir -p "logs/_archive/${TIMESTAMP}"
    mv "$logfile" "logs/_archive/${TIMESTAMP}/${slug}.log"
  fi
  mkdir -p "$resultdir"

  # Fresh working copy, built fresh on the node it will run on.
  rm -rf "$workdir"
  cp -r fisk "$workdir"

  {
    echo "=== ${label} ==="
    echo "Start: $(date -Iseconds)"
  } > "$logfile"

  srun \
    --job-name="fisk-${slug}" \
    --exclusive \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --constraint="${constraint}" \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    --time="${TIME_LIMIT}" \
    bash -lc "
      set -e
      module load '${module}'
      export CC='${cc}'
      export CXX='${cxx}'
      cd '${workdir}'
      make
      ./bin/fisk_benchmarks --output-dir '../${resultdir}'
    " >> "$logfile" 2>&1
  local rc=$?

  {
    echo "End:   $(date -Iseconds)"
    if [[ $rc -eq 0 ]]; then
      echo "Result: PASS"
    else
      echo "Result: FAIL (exit code ${rc})"
    fi
  } >> "$logfile"

  echo "$rc" > "${STATUS_DIR}/${slug}"
}

# ------------------------------------------------------------------------
#   Launch all combos in parallel.
# ------------------------------------------------------------------------

for combo in "${COMBOS[@]}"; do
  arch="${combo%%:*}"
  compiler="${combo##*:}"
  run_combo "$arch" "$compiler" &
done
wait

# ------------------------------------------------------------------------
#   Summary
# ------------------------------------------------------------------------

echo
echo "=================================================================="
echo " Summary"
echo "=================================================================="
overall_rc=0
for combo in "${COMBOS[@]}"; do
  arch="${combo%%:*}"
  compiler="${combo##*:}"
  slug="${arch}-${compiler}"
  label="${CPU_LABEL[$arch]}, ${COMPILER_LABEL[$compiler]}"
  rc="$(cat "${STATUS_DIR}/${slug}" 2>/dev/null || echo "?")"
  if [[ "$rc" == "0" ]]; then
    printf "  %-45s OK      logs/%s.log\n" "$label" "${slug}.log"
  else
    printf "  %-45s FAILED  logs/%s.log\n" "$label" "${slug}.log"
    overall_rc=1
  fi
done
echo "=================================================================="

exit "$overall_rc"
