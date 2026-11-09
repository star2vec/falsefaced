#!/usr/bin/env bash
# Start the local notebook session: JupyterLab (localhost only, random port + token) from the project env,
# in the background, with its Jupyter state under .nb/. Writes .nb/current.env, which scripts/nb_mcp.sh reads.
# Mac version of the template's nb_start.sh (no Slurm, no tunnel).   scripts/nb_start.sh [--port <port>]
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=""
while [ $# -gt 0 ]; do case "$1" in
  --port) PORT=$2; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done
mkdir -p .nb
ENVF=.nb/current.env
if [ -f "$ENVF" ]; then
  . "$ENVF"
  if kill -0 "$NB_PID" 2>/dev/null; then
    echo "session already running (pid $NB_PID): $JUPYTER_URL/lab?token=$JUPYTER_TOKEN"; exit 0
  fi
fi
PY=.venv/bin
TOKEN=$($PY/python -c "import secrets; print(secrets.token_hex(16))")
[ -n "$PORT" ] || PORT=$($PY/python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
# Per-session Jupyter data dir, as in the template, so this server never shares sqlite state with another one.
export JUPYTER_DATA_DIR=$PWD/.nb/jupyter-current
nohup $PY/jupyter lab --no-browser --ip 127.0.0.1 --port "$PORT" \
  --IdentityProvider.token "$TOKEN" --ServerApp.root_dir "$PWD" >.nb/current.log 2>&1 &
PID=$!
cat > "$ENVF" <<ENV
NB_PID=$PID
NB_HOST=$(hostname)
NB_PORT=$PORT
JUPYTER_URL=http://127.0.0.1:$PORT
JUPYTER_TOKEN=$TOKEN
ENV
for i in $(seq 1 60); do
  curl -sS -m 2 "http://127.0.0.1:$PORT/api/status?token=$TOKEN" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "jupyter lab died; see .nb/current.log"; exit 1; }
  sleep 0.5
done
echo "JupyterLab (pid $PID):  http://127.0.0.1:$PORT/lab?token=$TOKEN"
echo "Stop with: scripts/nb_stop.sh"
