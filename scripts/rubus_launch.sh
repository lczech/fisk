#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#   Fisk benchmark launcher for the rubus cluster
# ==============================================================================
#
# Starts scripts/submit_rubus_benchmarks.sh from a new fisk.zip, see scripts/make_fisk_zip.sh.
#
# Copy this script once into the working directory on the cluster, where fisk.zip goes. After
# each copy of a new fisk.zip there, run it from there, in tmux or screen, as the submission waits
# for its jobs:
#
#     ./rubus_launch.sh
#
# It replaces fisk/ with the contents of fisk.zip, gets the human genome into genome/ once (see
# example/run_human_genome.sh), and runs the submit_rubus_benchmarks.sh of the new zip with it.
# It does nothing else, so that a copy of it keeps working with newer zips.

cd "$(dirname "$0")"
if [[ ! -f fisk.zip ]]; then
  echo "Error: no fisk.zip in $(pwd)." >&2
  exit 1
fi

rm -rf fisk
unzip -q fisk.zip

FISK_GENOME_FILE="$(fisk/example/run_human_genome.sh --data-dir "$(pwd)/genome" --download-only)"
export FISK_GENOME_FILE
exec bash fisk/scripts/submit_rubus_benchmarks.sh
