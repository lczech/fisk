#!/usr/bin/env bash
set -euo pipefail

# Group matching plots by filename, naming copies after their CPU and compiler, so that one plot
# can be compared across all of them.
# Example: results/Intel_Core_7_240H/GCC_15/sum/kmer_extract_ascii_supp_sum_ops_log.png ->
# results/Collect/kmer_extract_ascii_supp_sum_ops_log/Intel_Core_7_240H_GCC_15.png.
cd "$(git rev-parse --show-toplevel)"
source plot/log.sh

# --------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <filename|--all>" >&2
    exit 1
fi

selection="$1"
if [[ "$selection" != "--all" ]]; then
    case "$selection" in
        */*|.*) error "Expected a plot filename, without a directory."; exit 1 ;;
    esac
    case "$selection" in
        *.png|*.pdf|*.svg) ;;
        *) error "Expected a PNG, PDF, or SVG filename."; exit 1 ;;
    esac
fi

indir="results"
outdir="${indir}/Collect"
shopt -s nullglob

# --------------------------------------------------------------------
# Discover sources
# --------------------------------------------------------------------

# The runs, results/<CPU>/<Compiler>/<sink>/, and their labels <CPU>_<Compiler>. The plots of the
# sinks of one CPU and compiler have the sink in their names, so they share the label.
sources=()
destinations=()
run_dirs=()
run_labels=()

for info in "$indir"/*/*/*/sys_info.txt; do
    run_dir="${info%/sys_info.txt}"
    compiler_dir="${run_dir%/*}"
    label="${compiler_dir#"${indir}/"}"
    label="${label/\//_}"
    run_dirs+=("$run_dir")
    run_labels+=("$label")

    for file in "$run_dir"/*.png "$run_dir"/*.pdf "$run_dir"/*.svg; do
        fname="${file##*/}"
        [[ "$selection" == "--all" || "$fname" == "$selection" ]] || continue
        sources+=("$file")
        destinations+=("${outdir}/${fname%.*}/${label}.${fname##*.}")
    done
done

# --------------------------------------------------------------------
# Copy current plots and report retained stale files
# --------------------------------------------------------------------

for ((i = 0; i < ${#sources[@]}; ++i)); do
    mkdir -p "${destinations[i]%/*}"
    cp -- "${sources[i]}" "${destinations[i]}"
done

stale_count=0
for file in "$outdir"/*/*.png "$outdir"/*/*.pdf "$outdir"/*/*.svg; do
    if [[ "$selection" != "--all" ]]; then
        [[ "${file%/*}" == "$outdir/${selection%.*}" && "${file##*.}" == "${selection##*.}" ]] || continue
    fi
    refreshed=false
    plot_dir="${file%/*}"
    fname="${plot_dir##*/}.${file##*.}"
    for ((i = 0; i < ${#run_dirs[@]}; ++i)); do
        if [[ "${file##*/}" == "${run_labels[i]}.${file##*.}" && -f "${run_dirs[i]}/$fname" ]]; then
            refreshed=true
            break
        fi
    done
    if [[ "$refreshed" == false ]]; then
        warn "retained without a current source: $file"
        stale_count=$((stale_count + 1))
    fi
done

echo "Collected ${#sources[@]} files; retained $stale_count files without a current source."
