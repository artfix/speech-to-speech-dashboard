#!/usr/bin/env bash
# Launch the speech-to-speech web dashboard on Linux / macOS.
#
# This script:
#   1. Ensures uv is installed (https://docs.astral.sh/uv/).
#   2. Runs `uv sync` on first run to install dependencies.
#   3. Starts the dashboard at http://localhost:8050 and opens it in your browser.
#
# The dashboard wraps the existing `speech-to-speech` CLI. It does NOT
# modify any code in src/ -- it spawns the pipeline as a subprocess.

set -euo pipefail

# Move to the script's own directory so paths work regardless of CWD.
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo ""
    echo "ERROR: 'uv' is not installed."
    echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or see:          https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    exit 1
fi

# Install dependencies only when the lockfile or pyproject has changed
# since the last sync. `uv sync` re-resolves the dependency graph every
# time it runs, and pyproject.toml pins ``torch>=2.4.0`` (no upper bound)
# so an unconditional sync can pull a different torch + matching NVIDIA
# wheels on every launch. We only want the resolver to run when the
# project actually changed -- first clone, after ``git pull``, or after
# the user edits pyproject/uv.lock by hand.
NEED_SYNC=1
if [ -d ".venv" ] && [ -f "uv.lock" ]; then
    # uv stamps ``uv.lock`` into .venv's metadata on every successful
    # sync. Compare the stamp against the on-disk lockfile.
    STAMP_FILE=".venv/.lock-stamp"
    if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE" 2>/dev/null)" = "$(stat -c %Y uv.lock 2>/dev/null)" ]; then
        NEED_SYNC=0
    fi
fi
if [ "$NEED_SYNC" = "1" ]; then
    uv sync
    # Record the lockfile mtime so the next launch can skip the sync.
    if [ -f "uv.lock" ]; then
        mkdir -p .venv
        stat -c %Y uv.lock > .venv/.lock-stamp 2>/dev/null || true
    fi
fi

URL="http://localhost:8050"

echo ""
echo "==========================================="
echo "  speech-to-speech dashboard"
echo "  opening ${URL}"
echo "  (close this terminal to stop the server)"
echo "==========================================="
echo ""

# Best-effort browser auto-open. macOS uses `open`, Linux `xdg-open`,
# WSL `wslview`. Failures are silently ignored -- headless boxes and
# SSH-only sessions will just see the URL printed.
(
    sleep 2
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "${URL}" >/dev/null 2>&1 &
    elif command -v open    >/dev/null 2>&1; then open    "${URL}" >/dev/null 2>&1 &
    elif command -v wslview >/dev/null 2>&1; then wslview "${URL}" >/dev/null 2>&1 &
    fi
) &

# Stash the original launch command so the auto-restart hook after a
# torch wheel swap can re-launch us with the same flags (NOT
# `uv run` plain, which would re-`uv sync` and clobber the install).
export SPEECH_TO_SPEECH_DASHBOARD_RESTART_CMD='exec uv run --no-sync speech-to-speech-web'

exec uv run --no-sync speech-to-speech-web
