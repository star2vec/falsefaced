#!/usr/bin/env bash
# Verify the running notebook session end to end: JupyterLab answers, and the jupyter MCP server
# starts and lists its tools.
set -euo pipefail
cd "$(dirname "$0")/.."
ENVF=.nb/current.env; [ -f "$ENVF" ] || { echo "no session; run scripts/nb_start.sh"; exit 1; }
. "$ENVF"
kill -0 "$NB_PID" 2>/dev/null && echo "JupyterLab pid $NB_PID: alive" || { echo "JupyterLab pid $NB_PID: NOT running"; exit 1; }
curl -sS -m 10 "$JUPYTER_URL/api/status?token=$JUPYTER_TOKEN" >/dev/null && echo "JupyterLab: ok at $JUPYTER_URL" || { echo "JupyterLab: NOT answering"; exit 1; }
scripts/mcp_call.py list || { echo "MCP server: FAILED"; exit 1; }
