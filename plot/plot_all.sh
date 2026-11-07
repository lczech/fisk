#!/usr/bin/env bash
set -euo pipefail

# Make all plots of the results:
#  - the plots of each benchmark run, in its folder results/<CPU>/<Compiler>/<sink>/ (plot_run.sh),
#  - the plots across CPUs and compilers, for each sink mode, in results/Summaries/
#    (plot_summaries.sh, which has the list of CPUs to show),
#  - copies of the plots of the runs, grouped by plot, in results/Collect/ (collect.sh).

# Change to top level of git repo.
# This ensures that the script can be called from any directory.
cd "$(git rev-parse --show-toplevel)"
source plot/log.sh

[[ -d results ]] || { warn "No results found in results/"; exit 0; }

./plot/plot_run.sh
for sink in sum barrier write; do
  echo
  echo "Plotting the summaries of ${sink}"
  ./plot/plot_summaries.sh "$sink"
done
echo
./plot/collect.sh --all
