#!/usr/bin/env bash
# MCP launcher used by .mcp.json: reads the running notebook session and starts jupyter-mcp-server
# against it over stdio. jupyter-mcp-server lives in its own env (uv tool), not the project env.
set -euo pipefail
cd "$(dirname "$0")/.."
ENVF=.nb/current.env
if [ ! -f "$ENVF" ]; then
  echo "no notebook session running; start one with scripts/nb_start.sh" >&2; exit 1
fi
. "$ENVF"
export JUPYTER_URL JUPYTER_TOKEN ALLOW_IMG_OUTPUT=true
exec "$HOME/.local/bin/jupyter-mcp-server" --transport stdio
