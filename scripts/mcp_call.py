#!/usr/bin/env python3
"""Drive scripts/nb_mcp.sh over stdio JSON-RPC from the shell (the server does not exit on stdin EOF).
  scripts/mcp_call.py list                      -> tool names
  scripts/mcp_call.py call <tool> '<json args>'  [<tool> '<json args>' ...]  -> each result
"""
import json, os, subprocess, sys, threading, queue

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = subprocess.Popen([os.path.join(root, "scripts", "nb_mcp.sh")], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=root)
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l) for l in p.stdout], daemon=True).start()

def send(msg):
    p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()

def recv(mid, timeout=180):
    while True:
        l = q.get(timeout=timeout).strip()
        if not l.startswith("{"): continue
        m = json.loads(l)
        if m.get("id") == mid: return m

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05",
      "capabilities": {}, "clientInfo": {"name": "mcp_call", "version": "0"}}})
recv(1)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
rc = 0
try:
    if sys.argv[1] == "list":
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        recv_tools = recv(2)["result"]["tools"]; t = [x["name"] for x in recv_tools]
        print(f"MCP server: ok, {len(t)} tools:", ", ".join(t))
        if "-v" in sys.argv:
            for x in recv_tools:
                print(x["name"], json.dumps(x["inputSchema"].get("properties", {})))
    else:
        args = sys.argv[2:]; mid = 2
        for name, a in zip(args[::2], args[1::2]):
            send({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                  "params": {"name": name, "arguments": json.loads(a)}})
            r = recv(mid); mid += 1
            print(f"== {name}")
            if "error" in r: print("ERROR", json.dumps(r["error"])); rc = 1
            else:
                for c in r["result"].get("content", []):
                    print(c.get("text", json.dumps(c)))
                if r["result"].get("isError"): rc = 1
except Exception as e:
    print("FAILED:", e); rc = 1
    try: print(p.stderr.read(4000))
    except Exception: pass
finally:
    p.kill()
sys.exit(rc)
