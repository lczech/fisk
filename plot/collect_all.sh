#!/usr/bin/env bash
set -euo pipefail

# Collect every plot from named CPU/compiler runs.
cd "$(git rev-parse --show-toplevel)"
exec ./plot/collect.sh --all
