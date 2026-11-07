#!/usr/bin/env bash
#
# Download a human genome, build the example, and run it on the genome, see README.md.
#
# Without options: GRCh38, downloaded into example/data/ and unpacked there, where later runs reuse
# it, and built with the default C++ compiler of CMake, in example/build/. Requires CMake (3.20 or
# later), a C++20 compiler, curl, and about 5 GB of disk space and 4 GB of memory. Written for
# bash 3.2, which macOS ships.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: example/run_human_genome.sh [options]

Downloads a human genome, builds the example, and runs it on the genome, printing a CSV row per
extractor.

Options:
  --genome <name>       grch38: GRCh38, no-alt analysis set (default);
                        t2t: T2T-CHM13v2.0.
  --genome-file <path>  A FASTA file to use instead of the download, plain or gzipped.
  --compiler <name>     C++ compiler to build with, e.g. g++ or clang++. Default: that of CMake.
  --cpu <n>             Run on this CPU only, pinned via taskset (Linux).
  --output <file>       Write the CSV to this file instead of the terminal.
  -h, --help            Show this help.
EOF
}

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
GENOME="grch38"
GENOME_FILE=""
COMPILER=""
CPU=""
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --genome|--genome-file|--compiler|--cpu|--output)
            if [ $# -lt 2 ]; then
                echo "Error: $1 needs a value." >&2
                exit 1
            fi
            ;;
    esac
    case "$1" in
        --genome)      GENOME="$2"; shift 2 ;;
        --genome-file) GENOME_FILE="$2"; shift 2 ;;
        --compiler)    COMPILER="$2"; shift 2 ;;
        --cpu)         CPU="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "Error: unknown option '$1'." >&2; usage >&2; exit 1 ;;
    esac
done

# ------------------------------------------------------------------------
#   Genome
# ------------------------------------------------------------------------

# The genomes, from NCBI, with the SHA-256 of their files.
NCBI="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA"
case "$GENOME" in
    grch38)
        URL="${NCBI}/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"
        SHA256="fb4243ebb014caf27111f24dd62b7ce42160f28581da6f8fcd6cba5977778d02"
        ;;
    t2t)
        URL="${NCBI}/009/914/755/GCA_009914755.4_T2T-CHM13v2.0/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna.gz"
        SHA256="a2f8b712a1958a41b590ce67743c324d64462e17b6b4fdc62747ff4de4b09c75"
        ;;
    *)
        echo "Error: unknown genome '$GENOME'; grch38 or t2t." >&2
        exit 1
        ;;
esac

sha256() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

DATA="${EXAMPLE_DIR}/data"
mkdir -p "$DATA"
if [ -z "$GENOME_FILE" ]; then
    GENOME_FILE="${DATA}/$(basename "$URL")"
    if [ ! -f "$GENOME_FILE" ]; then
        echo "Downloading $(basename "$URL")" >&2
        curl -fL --retry 3 -o "${GENOME_FILE}.part" "$URL"
        if [ "$(sha256 "${GENOME_FILE}.part")" != "$SHA256" ]; then
            echo "Error: the download does not have the expected checksum." >&2
            rm -f "${GENOME_FILE}.part"
            exit 1
        fi
        mv "${GENOME_FILE}.part" "$GENOME_FILE"
    fi
elif [ ! -f "$GENOME_FILE" ]; then
    echo "Error: genome file '$GENOME_FILE' not found." >&2
    exit 1
fi

# The example reads plain FASTA, so a gzipped file is unpacked into data/, once.
FASTA="$GENOME_FILE"
case "$GENOME_FILE" in
    *.gz)
        FASTA="${DATA}/$(basename "${GENOME_FILE%.gz}")"
        if [ ! -f "$FASTA" ]; then
            echo "Unpacking $(basename "$GENOME_FILE")" >&2
            gunzip -c "$GENOME_FILE" > "${FASTA}.part"
            mv "${FASTA}.part" "$FASTA"
        fi
        ;;
esac

# ------------------------------------------------------------------------
#   Build and run
# ------------------------------------------------------------------------

BUILD="${EXAMPLE_DIR}/build/$(basename "${COMPILER:-default}")"
CMAKE_ARGS=()
if [ -n "$COMPILER" ]; then
    CMAKE_ARGS=( -DCMAKE_CXX_COMPILER="$COMPILER" )
fi
echo "Building the example in ${BUILD}" >&2
cmake -S "$EXAMPLE_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    ${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"} > /dev/null
cmake --build "$BUILD" > /dev/null

PIN=()
if [ -n "$CPU" ]; then
    if ! command -v taskset > /dev/null 2>&1; then
        echo "Error: --cpu needs taskset, which is not available." >&2
        exit 1
    fi
    PIN=( taskset -c "$CPU" )
fi

echo "Running the example on $(basename "$FASTA")" >&2
if [ -n "$OUTPUT" ]; then
    mkdir -p "$(dirname "$OUTPUT")"
    ${PIN[@]+"${PIN[@]}"} "${BUILD}/fisk_example" "$FASTA" > "$OUTPUT"
else
    ${PIN[@]+"${PIN[@]}"} "${BUILD}/fisk_example" "$FASTA"
fi
