#!/usr/bin/env bash
set -uo pipefail

# ==============================================================================
#   Fisk source packager
# ==============================================================================
#
# Zips up everything needed to compile fisk (library headers, benchmarks,
# tests, mask files, and the cmake/make setup) into fisk.zip at the repo
# root, wrapped in a top-level fisk/ folder. Intended for scp'ing to a
# cluster to build and run there; grabs whatever is currently on disk,
# including uncommitted local changes.
#
# Usage (from anywhere inside the repo):
#
#     ./scripts/make_fisk_zip.sh

# ------------------------------------------------------------------------
#   Setup
# ------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" || ! -f "${REPO_ROOT}/Makefile" ]]; then
  echo "Error: must be run from inside the fisk git checkout." >&2
  exit 1
fi

if ! command -v zip &>/dev/null; then
  echo "Error: 'zip' is not installed." >&2
  exit 1
fi

cd "${REPO_ROOT}"

OUT_ZIP="${REPO_ROOT}/fisk.zip"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

# ------------------------------------------------------------------------
#   Stage files
# ------------------------------------------------------------------------

PKG_DIR="${STAGE_DIR}/fisk"
mkdir -p "${PKG_DIR}"

cp CMakeLists.txt Makefile README.md LICENSE "${PKG_DIR}/"
cp -r include benchmarks tests masks "${PKG_DIR}/"

# ------------------------------------------------------------------------
#   Zip it up
# ------------------------------------------------------------------------

rm -f "${OUT_ZIP}"
(cd "${STAGE_DIR}" && zip -rq "${OUT_ZIP}" fisk)

echo "Wrote ${OUT_ZIP} ($(du -h "${OUT_ZIP}" | cut -f1))"
