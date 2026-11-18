#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
#   Fisk benchmark submission for the rubus cluster
# ==============================================================================
#
# Runs scripts/run_benchmarks.sh, with all benchmark sink modes, on its own node for each
# combination of CPU arch x compiler:
#   - AMD EPYC 9684X (Zen 4)  x  Clang 21 / GCC 15
#   - AMD EPYC 9655 (Zen 5)   x  Clang 21 / GCC 15
#   - Intel Xeon 8568Y        x  Clang 21 / GCC 15
#
# Usage:
#   Run from a working directory that already contains a `fisk/` subdirectory
#   with exactly the source tree to build - either a fresh `git clone`, or an
#   unzipped copy made by scripts/make_fisk_zip.sh.
#
#     ./fisk/scripts/submit_rubus_benchmarks.sh [--shared]
#
# By default, each job has its node to itself (--exclusive). With --shared, each job takes
# SHARED_CORES whole cores (both threads of each) of a node that is idle when the script starts,
# a different one per job, and leaves the rest of the node to other jobs. The benchmarks then run
# on one of these cores, but other jobs on the node share the memory bandwidth and the power
# budget of the CPU with them. So the jobs log the CPUs allocated on their node, and its load, once
# a minute into ./output/logs/<arch>-<compiler>.occupancy.txt; the CPUs of the job itself count.
#
# With FISK_GENOME_FILE set to a copy of the human genome, the example uses it instead of
# downloading the genome on each node, see example/run_human_genome.sh. scripts/rubus_launch.sh
# does that, and unpacks the zip: copy it once next to fisk.zip, and run it after each new zip.
#
# The script waits for the jobs, so run it in tmux or screen.
#
# The jobs are submitted with srun, all at once, and the script waits for them. Everything
# lands in ./output/, so that one folder holds all runs: results in
# ./output/results/<CPU>/<Compiler>_<version>/<mode>/, as with run_benchmarks.sh, which moves
# earlier results of the same CPU, compiler and mode to ./output/results/_archive/; logs in
# ./output/logs/, where earlier logs are archived as well.

# ------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------

ACCOUNT="spear"
QOS="normal"
TIME_LIMIT="4:00:00"

# CPUs per job: all of them build, and the benchmarks run pinned to one.
CPUS_PER_TASK=64

# With --shared: whole cores per job (--hint=nomultithread, which allocates both threads of each
# core, while Slurm counts the cores), the memory, and a time limit for the slower build.
SHARED_CORES=8
SHARED_MEM="64G"
SHARED_TIME_LIMIT="6:00:00"

SHARED=0
for arg in "$@"; do
  case "$arg" in
    --shared) SHARED=1 ;;
    *) echo "Error: unknown argument '${arg}'." >&2; exit 1 ;;
  esac
done

# A copy of the human genome for the example, which run_benchmarks.sh runs on it, see
# example/run_human_genome.sh: used instead of the download if set, see the usage above.
GENOME_FILE="${FISK_GENOME_FILE:-}"

# Node constraints, per CPU arch. "std" excludes the gpu/hgx nodes.
declare -A CONSTRAINT=(
  [epyc]="std,epyc,znver4"
  [zen5]="std,epyc,znver5"
  [xeon]="std,xeon,sapphirerapids"
)

# Module to load, per compiler, and a folder to put in front of the PATH, for compilers without a
# module that loads: the module of Clang 21 (LLVM/21.1.8-GCCcore-15.2.0) asks for two versions of
# zlib, so Clang 21 runs from its folder, on top of the GCC 15.2 module, whose libstdc++ it uses.
declare -A MODULE=(
  [clang]="GCC/15.2.0"
  [gcc]="GCC/15.2.0"
)
declare -A COMPILER_PATH=(
  [clang]="/opt/software/software/LLVM/21.1.8-GCCcore-15.2.0/bin"
  [gcc]=""
)

# CPU names for the results/<CPU>/ folders, as the CPU model strings of these nodes are not the
# usual names; run_benchmarks.sh replaces the spaces by underscores.
declare -A CPU_LABEL=(
  [epyc]="AMD EPYC 9684X"
  [zen5]="AMD EPYC 9655"
  [xeon]="Intel Xeon Platinum 8568Y"
)

# The combos to run. Each gets its own node, so all of them run fully in parallel with no timing
# interference between them.
COMBOS=(
  "epyc:clang"
  "epyc:gcc"
  "zen5:clang"
  "zen5:gcc"
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
OUTPUT_DIR="$(pwd)/output"
RESULTS_ROOT="${OUTPUT_DIR}/results"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# With --shared: an idle node per combo, with all features of its constraint, each node once.
declare -A NODE=()
if [[ $SHARED -eq 1 ]]; then
  idle_nodes="$(sinfo -N -h -t idle -o '%n %f')"
  used_nodes=" "
  for combo in "${COMBOS[@]}"; do
    arch="${combo%%:*}"
    slug="${arch}-${combo##*:}"
    while read -r node features; do
      [[ -n "$node" && "$used_nodes" != *" ${node} "* ]] || continue
      fits=1
      for feature in ${CONSTRAINT[$arch]//,/ }; do
        [[ ",${features}," == *",${feature},"* ]] || fits=0
      done
      if [[ $fits -eq 1 ]]; then
        NODE[$slug]="$node"
        used_nodes+="${node} "
        break
      fi
    done <<< "$idle_nodes"
    if [[ -z "${NODE[$slug]:-}" ]]; then
      echo "Error: no idle node left for ${slug} (${CONSTRAINT[$arch]})." >&2
      exit 1
    fi
  done
fi

# ------------------------------------------------------------------------
#   One combo: fresh copy, then build and run all modes on one node.
# ------------------------------------------------------------------------

run_combo() {
  local arch="$1" compiler="$2"
  local slug="${arch}-${compiler}"
  local constraint="${CONSTRAINT[$arch]}"
  local module="${MODULE[$compiler]}"
  local compiler_path="${COMPILER_PATH[$compiler]}"
  local cpu_label="${CPU_LABEL[$arch]}"
  local workdir="fisk-${slug}"
  local logfile="${LOG_DIR}/${slug}.log"
  local genome_arg=""
  if [[ -n "$GENOME_FILE" ]]; then
    genome_arg="--genome-file '${GENOME_FILE}'"
  fi

  local occupancy="${LOG_DIR}/${slug}.occupancy.txt"

  # Archive any previous log of this combo instead of overwriting it.
  for f in "$logfile" "$occupancy"; do
    if [[ -f "$f" ]]; then
      mkdir -p "${LOG_DIR}/_archive/${TIMESTAMP}"
      mv "$f" "${LOG_DIR}/_archive/${TIMESTAMP}/"
    fi
  done

  # Fresh working copy, built fresh on the node it will run on.
  rm -rf "$workdir"
  cp -r fisk "$workdir"

  # Exclusive: --mem=0 asks for all memory of the node, without which Slurm refuses exclusive EPYC
  # nodes. Shared: the node picked above, whole cores, and the occupancy log, in the background
  # for as long as the job runs. The builds use all threads of the job, $(nproc) on the node.
  local alloc=(--exclusive --mem=0 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}")
  local build_jobs="${CPUS_PER_TASK}"
  local monitor=""
  if [[ $SHARED -eq 1 ]]; then
    alloc=(
      --nodelist="${NODE[$slug]}" --mem="${SHARED_MEM}" --cpus-per-task="${SHARED_CORES}"
      --hint=nomultithread --time="${SHARED_TIME_LIMIT}"
    )
    build_jobs='$(nproc)'
    monitor="
      while true; do
        echo \"\$(date -Iseconds) \$(scontrol show node \"\$(hostname -s)\" \
          | grep -oE 'CPUAlloc=[0-9]+|CPULoad=[0-9.]+' | tr '\n' ' ')\"
        sleep 60
      done >> '${occupancy}' &
      trap \"kill \$! 2> /dev/null || true\" EXIT
    "
  fi

  {
    echo "=== ${slug} ==="
    echo "Start: $(date -Iseconds)"
    if [[ $SHARED -eq 1 ]]; then echo "Shared: ${NODE[$slug]}, ${SHARED_CORES} cores"; fi
  } > "$logfile"

  srun \
    --job-name="fisk-${slug}" \
    "${alloc[@]}" \
    --nodes=1 --ntasks=1 \
    --constraint="${constraint}" \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    bash -lc "
      set -e
      ${monitor}
      module load '${module}'
      ${compiler_path:+export PATH=\"${compiler_path}:\$PATH\"}
      cd '${workdir}'
      scripts/run_benchmarks.sh --compilers '${compiler}' --cpu-label '${cpu_label}' \
        --jobs ${build_jobs} --results-dir '${RESULTS_ROOT}' ${genome_arg}
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
    printf "  %-45s OK      output/logs/%s.log\n" "${CPU_LABEL[$arch]}, ${compiler}" "${slug}"
  else
    printf "  %-45s FAILED  output/logs/%s.log\n" "${CPU_LABEL[$arch]}, ${compiler}" "${slug}"
    overall_rc=1
  fi
done
echo "=================================================================="

exit "$overall_rc"
