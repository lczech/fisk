# Messages of the plot scripts on stderr, sourced by them: warn() for something that is left out,
# such as a missing result table, and error() for something that stops the script. The prefix is
# colored when stderr is a terminal, and plain in logs. plot_common.py has the same warn() for the
# Python plot scripts.

if [[ -t 2 ]]; then
  LOG_WARN_PREFIX=$'\033[1;33mWARNING:\033[0m'
  LOG_ERROR_PREFIX=$'\033[1;31mERROR:\033[0m'
else
  LOG_WARN_PREFIX="WARNING:"
  LOG_ERROR_PREFIX="ERROR:"
fi

warn() {
  echo "${LOG_WARN_PREFIX} $*" >&2
}

error() {
  echo "${LOG_ERROR_PREFIX} $*" >&2
}
