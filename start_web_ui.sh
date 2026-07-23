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

# Install dependencies on first run (or after a pyproject.toml change).
uv sync

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

exec uv run speech-to-speech-web
