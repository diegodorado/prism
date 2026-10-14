#!/usr/bin/env bash
# Headless StreamDiffusion server launcher (Linux, native venv).
#
# Usage:
#   ./run.sh                      # uses configs/config.json
#   ./run.sh /path/to/cfg.json
#
# The server resolves models/ and models/model_configs relative to the current
# working directory, so we cd into StreamDiffusion/ (which also hosts .venv).
set -euo pipefail

# Keep child-process (multiprocessing spawn) log output flowing to the console.
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$ROOT/configs/config.json}"

cd "$ROOT/StreamDiffusion"
exec "$ROOT/StreamDiffusion/.venv/bin/python" "$ROOT/server/main.py" -c "$CONFIG"
