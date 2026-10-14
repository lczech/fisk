#!/usr/bin/env bash
set -euo pipefail

# Group matching plots by filename, naming copies after their CPU/compiler.
# Example: CPU/kmer_extract_ops_log.png -> Collect/kmer_extract_ops_log/CPU.png.
cd "$(git rev-parse --show-toplevel)"

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
        */*|.*) echo "Expected a plot filename, without a directory." >&2; exit 1 ;;
    esac
    case "$selection" in
        *.png|*.pdf|*.svg) ;;
        *) echo "Expected a PNG, PDF, or SVG filename." >&2; exit 1 ;;
    esac
fi

indir="results"
outdir="${indir}/Collect"
shopt -s nullglob

# --------------------------------------------------------------------
# Discover sources and check destination names
# --------------------------------------------------------------------

sources=()
destinations=()
cpu_dirs=()
cpu_labels=()

for info in "$indir"/*/sys_info.txt; do
    [[ -f "$info" ]] || continue
    cpu_dir="${info%/*}"
    case "${cpu_dir##*/}" in Collect|Summaries) continue ;; esac
    cpu="${cpu_dir##*/}"
    cpu="${cpu// /_}"

    # Reject ambiguous CPU labels before overwriting any collected file.
    for ((i = 0; i < ${#cpu_labels[@]}; ++i)); do
        if [[ "${cpu_labels[i]}" == "$cpu" ]]; then
            echo "Conflicting CPU labels: ${cpu_dirs[i]} and $cpu_dir" >&2
            exit 1
        fi
    done
    cpu_dirs+=("$cpu_dir")
    cpu_labels+=("$cpu")

    for file in "$cpu_dir"/*.png "$cpu_dir"/*.pdf "$cpu_dir"/*.svg; do
        [[ -f "$file" ]] || continue
        fname="${file##*/}"
        [[ "$selection" == "--all" || "$fname" == "$selection" ]] || continue
        destination="${outdir}/${fname%.*}/${cpu}.${fname##*.}"

        sources+=("$file")
        destinations+=("$destination")
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
    [[ -f "$file" ]] || continue
    if [[ "$selection" != "--all" ]]; then
        [[ "${file%/*}" == "$outdir/${selection%.*}" && "${file##*.}" == "${selection##*.}" ]] || continue
    fi
    refreshed=false
    plot_dir="${file%/*}"
    fname="${plot_dir##*/}.${file##*.}"
    for ((i = 0; i < ${#cpu_dirs[@]}; ++i)); do
        if [[ "${file##*/}" == "${cpu_labels[i]}.${file##*.}" && -f "${cpu_dirs[i]}/$fname" ]]; then
            refreshed=true
            break
        fi
    done
    if [[ "$refreshed" == false ]]; then
        echo "Warning: retained without a current source: $file" >&2
        stale_count=$((stale_count + 1))
    fi
done

echo "Collected ${#sources[@]} files; retained $stale_count files without a current source."
