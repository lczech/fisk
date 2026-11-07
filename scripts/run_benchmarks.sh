#!/usr/bin/env bash
#
# Build and run the benchmarks, for each compiler and each benchmark sink mode (sum, barrier,
# write; see benchmarks/sink.hpp), one after the other.
#
# Without options, this runs everything that is usual on the machine: GCC and Clang on Linux, as
# far as they are installed, and Apple Clang on macOS, each in all three modes. Each mode is built
# in its own folder in build-bench/, and then run on one CPU core, pinned via taskset on Linux.
# The results go to results/<CPU>/<Compiler>_<version>/<mode>/, e.g.
# results/Intel_Core_7_240H/GCC_15/sum/, with one CSV per suite, the system and benchmark settings
# in sys_info.txt, and the output in log.txt; names have underscores instead of spaces. Existing
# results of the same CPU, compiler and mode are moved to results/_archive/<date>/ first.
#
# Each build writes its benchmark program to bin/ under a name of its own, such as
# bin/fisk_benchmarks_gcc_15_sum, the same as its build folder build-bench/gcc_15_sum/.
#
# Requires CMake (3.20 or later) and a C++20 compiler. Written for bash 3.2, which macOS ships.

set -uo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_benchmarks.sh [options] [-- <benchmark options>]

Builds and runs the benchmarks for each compiler and each benchmark sink mode.

Options:
  --compilers <list>  Comma-separated compilers, e.g. gcc,clang or gcc-15. A C compiler name is
                      turned into its C++ driver (gcc -> g++, clang -> clang++).
                      Default: $CXX if set; else gcc and clang on Linux, as far as installed,
                      and the system compiler (Apple Clang) on macOS.
  --modes <list>      Comma-separated sink modes out of sum, barrier, write. Default: all three.
  --cpu <n>           CPU core to pin the runs to (Linux only). Default: 2, or the first allowed
                      CPU if 2 is not.
  --jobs <n>          Parallel build jobs. Default: the number of CPUs, at most one per GiB of
                      available memory.
  --cpu-label <name>  CPU name for the results folder. Default: the CPU model, shortened.
  --results-dir <dir> Folder for the results. Default: results/ in the repository.
  -h, --help          Show this help.

Everything after -- is passed to fisk_benchmarks, e.g. -- -k 31 --sample-ms 5.
See bin/fisk_benchmarks --help for those options.
EOF
}

# ------------------------------------------------------------------------
#   Options
# ------------------------------------------------------------------------

COMPILERS=""
MODES="sum barrier write"
CPU=""
JOBS=""
CPU_LABEL=""
RESULTS_ROOT=""
BENCH_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --compilers|--modes|--cpu|--jobs|--cpu-label|--results-dir)
            if [ $# -lt 2 ]; then
                echo "Error: option '$1' needs a value" >&2
                exit 1
            fi
            ;;
    esac
    case "$1" in
        --compilers) COMPILERS="${2//,/ }"; shift 2 ;;
        --modes)     MODES="${2//,/ }"; shift 2 ;;
        --cpu)       CPU="$2"; shift 2 ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --cpu-label) CPU_LABEL="$2"; shift 2 ;;
        --results-dir) RESULTS_ROOT="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; BENCH_ARGS=("$@"); break ;;
        *)           echo "Error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

for mode in $MODES; do
    case "$mode" in
        sum|barrier|write) ;;
        *) echo "Error: unknown mode '$mode'" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
OS="$(uname -s)"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"

if ! command -v cmake > /dev/null 2>&1; then
    echo "Error: cmake is not installed." >&2
    exit 1
fi

# ------------------------------------------------------------------------
#   Compilers and labels
# ------------------------------------------------------------------------

# Default compilers, as C++ drivers.
if [ -z "$COMPILERS" ]; then
    if [ -n "${CXX:-}" ]; then
        COMPILERS="$CXX"
    elif [ "$OS" = "Darwin" ]; then
        # gcc on macOS is another name for Apple Clang, so the system compiler is the only one.
        COMPILERS="c++"
    else
        for c in g++ clang++; do
            if command -v "$c" > /dev/null 2>&1; then
                COMPILERS="$COMPILERS $c"
            else
                echo "Note: $c not found, skipping it."
            fi
        done
    fi
fi
if [ -z "${COMPILERS// /}" ]; then
    echo "Error: no compiler found." >&2
    exit 1
fi

# C++ driver of a compiler name: gcc -> g++, gcc-15 -> g++-15, clang -> clang++.
cxx_driver() {
    case "$(basename "$1")" in
        *g++*|*clang++*|c++) echo "$1" ;;
        *clang*)             echo "${1/clang/clang++}" ;;
        *gcc*)               echo "${1/gcc/g++}" ;;
        *)                   echo "$1" ;;
    esac
}

# Compiler name and major version, e.g. "GCC 15", "Clang 21", "Apple Clang 17".
compiler_label() {
    local version_line major
    version_line="$("$1" --version 2>/dev/null | head -n 1)"
    major="$(echo "$version_line" | sed -E 's/.*version ([0-9]+).*/\1/')"
    case "$version_line" in
        *"Apple clang"*) echo "Apple Clang $major" ;;
        *clang*)         echo "Clang $major" ;;
        *)
            if "$1" --version 2>/dev/null | grep -q "Free Software Foundation"; then
                echo "GCC $("$1" -dumpversion | cut -d. -f1)"
            else
                echo "$(basename "$1") $("$1" -dumpversion 2>/dev/null | cut -d. -f1)"
            fi
            ;;
    esac
}

# CPU model string, shortened to the usual name, e.g.
# "INTEL(R) XEON(R) PLATINUM 8568Y+" -> "Intel Xeon Platinum 8568Y",
# "AMD EPYC 9684X 96-Core Processor" -> "AMD EPYC 9684X".
shorten_cpu_model() {
    local model pair
    model="$(echo "$1" | sed -E \
        -e 's/\((R|r|TM|tm)\)//g' \
        -e 's/ CPU @ .*//' \
        -e 's/ @ .*//' \
        -e 's/ [0-9]+-Core Processor//' \
        -e 's/ with .* Graphics//' \
        -e 's/\+ *$//' \
    )"
    model=" $(echo "$model" | tr -s ' ') "
    for pair in INTEL:Intel XEON:Xeon PLATINUM:Platinum GOLD:Gold SILVER:Silver CORE:Core PRO:Pro
    do
        model="$(echo "$model" | sed "s/ ${pair%%:*} / ${pair##*:} /g")"
    done
    echo "$model" | sed -E 's/^ +//; s/ +$//'
}

if [ -z "$CPU_LABEL" ]; then
    if [ "$OS" = "Darwin" ]; then
        cpu_model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    else
        cpu_model="$(grep -m 1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2-)"
    fi
    if [ -n "$cpu_model" ]; then
        CPU_LABEL="$(shorten_cpu_model "$cpu_model")"
    else
        CPU_LABEL="$(uname -m)"
    fi
fi

# ------------------------------------------------------------------------
#   Build jobs and CPU pinning
# ------------------------------------------------------------------------

if [ -z "$JOBS" ]; then
    if [ "$OS" = "Darwin" ]; then
        cpus="$(sysctl -n hw.ncpu)"
        mem_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    else
        cpus="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
        mem_gib="$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)"
    fi
    JOBS="$cpus"
    if [ -n "$mem_gib" ] && [ "$mem_gib" -lt "$JOBS" ]; then JOBS="$mem_gib"; fi
    if [ "$JOBS" -lt 1 ]; then JOBS=1; fi
fi

# Command prefix to pin the runs to one CPU: CPU 2 (or the given one) if this process may use it,
# else the first CPU it may use, as under Slurm or other restricted CPU sets.
PIN=()
if [ "$OS" != "Darwin" ]; then
    if command -v taskset > /dev/null 2>&1; then
        allowed=""
        for range in $(taskset -pc $$ | sed 's/.*: //' | tr ',' ' '); do
            first="${range%-*}"
            last="${range#*-}"
            i="$first"
            while [ "$i" -le "$last" ]; do allowed="$allowed $i"; i=$(( i + 1 )); done
        done
        wanted="${CPU:-2}"
        case " $allowed " in
            *" $wanted "*) pin_cpu="$wanted" ;;
            *)
                pin_cpu="$(echo "$allowed" | awk '{ print $1 }')"
                echo "Note: CPU $wanted is not available, pinning to CPU $pin_cpu instead."
                ;;
        esac
        PIN=(taskset -c "$pin_cpu")
    else
        echo "Note: taskset not found, running without pinning to a CPU."
    fi
fi

# ------------------------------------------------------------------------
#   Build and run
# ------------------------------------------------------------------------

echo "CPU:        $CPU_LABEL"
echo "Compilers:  $COMPILERS"
echo "Modes:      $MODES"
echo "Build jobs: $JOBS"
if [ ${#PIN[@]} -gt 0 ]; then echo "Pinning:    ${PIN[*]}"; fi

SUMMARY=""
FAILED=0

for compiler in $COMPILERS; do
    cxx="$(cxx_driver "$compiler")"
    if ! command -v "$cxx" > /dev/null 2>&1; then
        echo "Error: compiler '$cxx' not found." >&2
        SUMMARY="${SUMMARY}  ${cxx}: not found\n"
        FAILED=1
        continue
    fi
    clabel="$(compiler_label "$cxx")"
    label="${CPU_LABEL}, ${clabel}"
    slug="$(echo "$clabel" | tr 'A-Z ' 'a-z_')"
    # The results folder, e.g. Intel_Core_7_240H/GCC_15.
    result_name="$(echo "$CPU_LABEL" | tr ' ' '_')/$(echo "$clabel" | tr ' ' '_')"

    for mode in $MODES; do
        echo
        echo "=== ${label}, ${mode} ==="
        echo "Start: $(date +%Y-%m-%dT%H:%M:%S)"
        build_name="${slug}_${mode}"
        build_dir="${REPO_ROOT}/build-bench/${build_name}"
        result_dir="${RESULTS_ROOT}/${result_name}/${mode}"
        mode_define="FISK_BENCHMARK_SINK_$(echo "$mode" | tr 'a-z' 'A-Z')"

        # Build
        if ! cmake -S "$REPO_ROOT" -B "$build_dir" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_COMPILER="$cxx" \
                -DFISK_BUILD_TESTS=OFF \
                -DFISK_BENCHMARK_SINK_MODE="$mode_define" \
                -DFISK_BENCHMARK_NAME_SUFFIX="_${build_name}" > /dev/null \
            || ! cmake --build "$build_dir" --target fisk_benchmarks -j "$JOBS"
        then
            echo "Build failed: ${label}, ${mode}" >&2
            SUMMARY="${SUMMARY}  ${label}, ${mode}: build failed\n"
            FAILED=1
            continue
        fi

        # Run
        if [ -d "$result_dir" ]; then
            archive_dir="${RESULTS_ROOT}/_archive/${TIMESTAMP}/${result_name}"
            mkdir -p "$archive_dir"
            mv "$result_dir" "$archive_dir/"
            echo "Moved previous results to ${archive_dir}/${mode}"
        fi
        mkdir -p "$result_dir"
        ${PIN[@]+"${PIN[@]}"} "${REPO_ROOT}/bin/fisk_benchmarks_${build_name}" --output-dir "$result_dir" \
            ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} 2>&1 | tee "${result_dir}/log.txt"
        rc=${PIPESTATUS[0]}
        echo "End:   $(date +%Y-%m-%dT%H:%M:%S)"
        if [ "$rc" -eq 0 ]; then
            SUMMARY="${SUMMARY}  ${label}, ${mode}: OK\n"
        else
            echo "Run failed: ${label}, ${mode} (exit code ${rc})" >&2
            SUMMARY="${SUMMARY}  ${label}, ${mode}: run failed (exit code ${rc})\n"
            FAILED=1
        fi
    done
done

echo
echo "Summary:"
printf "%b" "$SUMMARY"
exit "$FAILED"
