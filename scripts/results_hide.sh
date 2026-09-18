#!/usr/bin/env bash
set -euo pipefail

# While actively developing, files under results/ change constantly (new benchmark runs, renamed
# columns, etc.), but we do not want to commit every intermediate version -- that would blow up
# the repo history with data we will regenerate anyway. At the same time, we do want to keep the
# files on disk as-is, since the plot scripts read straight from disk regardless of git state.
#
# This script hides that local churn from `git status`/`git diff`, without touching disk content
# or committing anything:
#   - Files already tracked by git (e.g. results/Summaries/*.csv) get `git update-index
#     --skip-worktree`, which tells git "ignore local differences to this file" while keeping its
#     last-committed content in the index.
#   - Files not yet tracked (new per-CPU benchmark CSVs, plots, etc.) get hidden via a `results/`
#     entry in .git/info/exclude, the local-only counterpart to .gitignore.
# Both are purely local git state -- nothing here is committed or shared. Run results_unhide.sh to
# reverse this (e.g. before committing a final set of results for a release).

cd "$(git rev-parse --show-toplevel)"

# Skip-worktree every currently tracked, locally modified/deleted file under results/.
git status --porcelain -- results/ \
    | grep -v '^??' \
    | cut -c4- \
    | sed 's/^"\(.*\)"$/\1/' \
    | while IFS= read -r f; do
        git update-index --skip-worktree -- "$f"
    done

# Hide untracked files under results/ via the local-only exclude file, idempotently.
EXCLUDE_FILE=".git/info/exclude"
if ! grep -qxF "results/" "$EXCLUDE_FILE" 2>/dev/null; then
    echo "results/" >> "$EXCLUDE_FILE"
fi

echo "results/ local changes hidden from git status/diff. Run plot/results_unhide.sh to reverse."
