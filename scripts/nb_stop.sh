#!/usr/bin/env bash
# Stop the local notebook session started by scripts/nb_start.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
ENVF=.nb/current.env
[ -f "$ENVF" ] || { echo "no session"; exit 0; }
. "$ENVF"
if kill -0 "$NB_PID" 2>/dev/null; then
  kill "$NB_PID"; sleep 1; kill -0 "$NB_PID" 2>/dev/null && kill -9 "$NB_PID" || true
  echo "stopped JupyterLab (pid $NB_PID)"
else
  echo "JupyterLab (pid $NB_PID) was not running"
fi
rm -f "$ENVF"
