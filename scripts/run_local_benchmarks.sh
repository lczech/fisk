#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
#   Fisk benchmark runner for the local machine
# ==============================================================================
#
# Runs the fisk benchmarks with both GCC and Clang, sequentially, in place in
# this git checkout (clean build + bin, rebuild, run).
#
# Usage (from anywhere inside the repo):
#
#     ./scripts/run_local_benchmarks.sh
#
# Results land in results/<CPU label>, <Compiler> <version>/, feeding into 
# plot_all_cpus.sh without renaming.

# ------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------

# This machine's CPU, matching the existing results/ naming convention.
CPU_LABEL="AMD Ryzen 7 Pro 4750U"

# Compilers to run, in order. Sequential (not parallel): same physical
# machine, so running both at once would contend for the same cores/cache
# and skew timings.
COMPILERS=(gcc clang)

declare -A CC_BIN=(  [gcc]="gcc"   [clang]="clang" )
declare -A CXX_BIN=( [gcc]="g++"   [clang]="clang++" )
declare -A COMPILER_NAME=( [gcc]="GCC" [clang]="Clang" )

# ------------------------------------------------------------------------
#   Setup
# ------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" || ! -f "${REPO_ROOT}/Makefile" ]]; then
  echo "Error: must be run from inside the fisk git checkout." >&2
  exit 1
fi
cd "$REPO_ROOT"

mkdir -p results

# ------------------------------------------------------------------------
#   One compiler: clean, build, run. Output streams straight to the
#   terminal - nothing here is worth keeping after the run.
# ------------------------------------------------------------------------

run_compiler() {
  local compiler="$1"
  local cc="${CC_BIN[$compiler]}"
  local cxx="${CXX_BIN[$compiler]}"
  local version
  version="$("$cc" -dumpversion | cut -d. -f1)"
  local label="${CPU_LABEL}, ${COMPILER_NAME[$compiler]} ${version}"
  local resultdir="results/${label}"

  mkdir -p "$resultdir"

  echo "=== ${label} ==="
  echo "Start: $(date -Iseconds)"

  local rc=0
  (
    set -e
    make clean
    CC="$cc" CXX="$cxx" make
    ./bin/fisk_benchmarks --output-dir "${resultdir}"
  ) || rc=$?

  echo "End:   $(date -Iseconds)"
  if [[ $rc -eq 0 ]]; then
    echo "Result: PASS"
  else
    echo "Result: FAIL (exit code ${rc})"
  fi

  return "$rc"
}

# ------------------------------------------------------------------------
#   Run sequentially, one compiler failing doesn't skip the other.
# ------------------------------------------------------------------------

declare -A RESULTS
overall_rc=0
for compiler in "${COMPILERS[@]}"; do
  echo "Running ${compiler}..."
  run_compiler "$compiler"
  rc=$?
  RESULTS[$compiler]="$rc"
  [[ "$rc" == "0" ]] || overall_rc=1
done

# ------------------------------------------------------------------------
#   Summary
# ------------------------------------------------------------------------

echo
echo "=================================================================="
echo " Summary"
echo "=================================================================="
for compiler in "${COMPILERS[@]}"; do
  rc="${RESULTS[$compiler]}"
  if [[ "$rc" == "0" ]]; then
    printf "  %-10s OK\n" "$compiler"
  else
    printf "  %-10s FAILED (exit code %s)\n" "$compiler" "$rc"
  fi
done
echo "=================================================================="

exit "$overall_rc"
