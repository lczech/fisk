#!/usr/bin/env bash
set -euo pipefail

# Reverses plot/results_hide.sh: restores normal git status/diff visibility for results/, so
# local changes show up again. Run this before committing a fresh set of results (e.g. for a
# release), then `git add`/`git status` as usual.

cd "$(git rev-parse --show-toplevel)"

# Un-skip-worktree every file currently marked skip-worktree under results/.
git ls-files -v -- results/ \
    | grep '^S ' \
    | cut -c3- \
    | while IFS= read -r f; do
        git update-index --no-skip-worktree -- "$f"
    done

# Remove the results/ entry from the local-only exclude file, if present.
EXCLUDE_FILE=".git/info/exclude"
if [ -f "$EXCLUDE_FILE" ] && grep -qxF "results/" "$EXCLUDE_FILE"; then
    grep -vxF "results/" "$EXCLUDE_FILE" > "${EXCLUDE_FILE}.tmp"
    mv "${EXCLUDE_FILE}.tmp" "$EXCLUDE_FILE"
fi

echo "results/ tracking restored to normal. New/changed files under results/ will show up in git status again."
