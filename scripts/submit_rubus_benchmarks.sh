#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
#   Fisk benchmark submission for the rubus cluster
# ==============================================================================
#
# Runs scripts/run_benchmarks.sh, with all benchmark sink modes, on its own exclusive node for
# each combination of CPU arch x compiler:
#   - AMD EPYC 9684X    x  Clang / GCC 15
#   - Intel Xeon 8568Y  x  Clang / GCC 15
#
# Usage:
#   Run from a working directory that already contains a `fisk/` subdirectory
#   with exactly the source tree to build - either a fresh `git clone`, or an
#   unzipped copy made by scripts/make_fisk_zip.sh.
#
#     ./fisk/scripts/submit_rubus_benchmarks.sh
#
# With FISK_GENOME_FILE set to a copy of the human genome, the example uses it instead of
# downloading the genome on each node, see example/run_human_genome.sh. With the zip, in an empty
# working directory:
#
#     unzip fisk.zip
#     NCBI=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38
#     GENOME=GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
#     curl -fLO "${NCBI}/seqs_for_alignment_pipelines.ucsc_ids/${GENOME}"
#     FISK_GENOME_FILE="${GENOME}" bash fisk/scripts/submit_rubus_benchmarks.sh
#
# The script waits for the jobs, so run it in tmux or screen.
#
# The jobs are submitted with srun, all four at once, and the script waits for them. Results land
# in ./results/<CPU>/<Compiler>_<version>/<mode>/, as with run_benchmarks.sh, which moves earlier
# results of the same CPU, compiler and mode to ./results/_archive/; logs in ./logs/, where
# earlier logs are archived as well.

# ------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------

ACCOUNT="spear"
QOS="normal"
TIME_LIMIT="4:00:00"

# CPUs per job: all of them build, and the benchmarks run pinned to one.
CPUS_PER_TASK=16

# A copy of the human genome for the example, which run_benchmarks.sh runs on it, see
# example/run_human_genome.sh: used instead of the download if set, see the usage above.
GENOME_FILE="${FISK_GENOME_FILE:-}"

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

# CPU names for the results/<CPU>/ folders, as the CPU model strings of these nodes are not the
# usual names; run_benchmarks.sh replaces the spaces by underscores.
declare -A CPU_LABEL=(
  [epyc]="AMD EPYC 9684X"
  [xeon]="Intel Xeon Platinum 8568Y"
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

# The jobs run in their own working copies, so the genome path has to be absolute.
if [[ -n "$GENOME_FILE" ]]; then
  if [[ ! -f "$GENOME_FILE" ]]; then
    echo "Error: FISK_GENOME_FILE '${GENOME_FILE}' does not exist." >&2
    exit 1
  fi
  GENOME_FILE="$(cd "$(dirname "$GENOME_FILE")" && pwd)/$(basename "$GENOME_FILE")"
fi

TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"
RESULTS_ROOT="$(pwd)/results"
mkdir -p "$RESULTS_ROOT" logs

STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# ------------------------------------------------------------------------
#   One combo: fresh copy, then build and run all modes on one node.
# ------------------------------------------------------------------------

run_combo() {
  local arch="$1" compiler="$2"
  local slug="${arch}-${compiler}"
  local constraint="${CONSTRAINT[$arch]}"
  local module="${MODULE[$compiler]}"
  local cpu_label="${CPU_LABEL[$arch]}"
  local workdir="fisk-${slug}"
  local logfile="logs/${slug}.log"
  local genome_arg=""
  if [[ -n "$GENOME_FILE" ]]; then
    genome_arg="--genome-file '${GENOME_FILE}'"
  fi

  # Archive any previous log for this combo instead of overwriting.
  if [[ -f "$logfile" ]]; then
    mkdir -p "logs/_archive/${TIMESTAMP}"
    mv "$logfile" "logs/_archive/${TIMESTAMP}/${slug}.log"
  fi

  # Fresh working copy, built fresh on the node it will run on.
  rm -rf "$workdir"
  cp -r fisk "$workdir"

  {
    echo "=== ${slug} ==="
    echo "Start: $(date -Iseconds)"
  } > "$logfile"

  srun \
    --job-name="fisk-${slug}" \
    --exclusive \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
    --constraint="${constraint}" \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    --time="${TIME_LIMIT}" \
    bash -lc "
      set -e
      module load '${module}'
      cd '${workdir}'
      scripts/run_benchmarks.sh --compilers '${compiler}' --cpu-label '${cpu_label}' \
        --jobs '${CPUS_PER_TASK}' --results-dir '${RESULTS_ROOT}' ${genome_arg}
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
  rc="$(cat "${STATUS_DIR}/${slug}" 2>/dev/null || echo "?")"
  if [[ "$rc" == "0" ]]; then
    printf "  %-45s OK      logs/%s.log\n" "${CPU_LABEL[$arch]}, ${compiler}" "${slug}.log"
  else
    printf "  %-45s FAILED  logs/%s.log\n" "${CPU_LABEL[$arch]}, ${compiler}" "${slug}.log"
    overall_rc=1
  fi
done
echo "=================================================================="

exit "$overall_rc"
