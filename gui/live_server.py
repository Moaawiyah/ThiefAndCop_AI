"""Live web GUI — watch the game in a browser at http://HOST:PORT (assignment §6).

Starts the background :class:`~gui.live_game.LiveGame` runner and a stdlib HTTP
server (no extra dependencies). ``GET /`` returns a small self-refreshing page;
``GET /state`` returns the current board + NL dialogue as JSON, which the page
polls every 500 ms to animate the pursuit and show the live trash-talk.

    python3 gui/live_server.py                 # then open the printed URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from gui.live_game import LiveGame

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Cop &amp; Thief — live</title><style>
body{background:#181a20;color:#e6e6e6;font-family:menlo,consolas,monospace;text-align:center}
#board{display:inline-grid;gap:4px;margin:16px auto}
.cell{width:64px;height:64px;background:#282b33;border-radius:8px;display:flex;
align-items:center;justify-content:center;font-size:26px;font-weight:bold;color:#fff}
.barrier{background:#787878}.cop{background:#4287f5}.thief{background:#eb5757}
#msgs{max-width:640px;margin:0 auto;text-align:left}
.cop-msg{color:#4287f5}.thief-msg{color:#eb5757}#hud{margin:10px;font-size:18px}
</style></head><body><h2>Cop &amp; Thief — live</h2>
<div id="hud"></div><div id="board"></div><div id="msgs"></div><script>
async function tick(){
  let s=await (await fetch('/state')).json();
  document.getElementById('hud').textContent=
    `sub-game ${s.sub_game}/${s.num_games}  move ${s.move}/${s.max_moves}  `+
    `cop ${s.totals.cop}  thief ${s.totals.thief}  — ${s.status}`;
  let b=document.getElementById('board');
  b.style.gridTemplateColumns=`repeat(${s.cols}, 64px)`; b.innerHTML='';
  let bar=new Set(s.barriers.map(p=>p[0]+','+p[1]));
  for(let r=0;r<s.rows;r++)for(let c=0;c<s.cols;c++){
    let d=document.createElement('div'); d.className='cell';
    if(s.cop[0]==r&&s.cop[1]==c){d.className='cell cop';d.textContent='C';}
    else if(s.thief[0]==r&&s.thief[1]==c){d.className='cell thief';d.textContent='T';}
    else if(bar.has(r+','+c)){d.className='cell barrier';}
    b.appendChild(d);
  }
  let m=document.getElementById('msgs'); m.innerHTML='';
  for(let [cls,txt] of [['cop-msg','COP: '+(s.cop_msg||'')],
                        ['thief-msg','THIEF: '+(s.thief_msg||'')]]){
    let p=document.createElement('p'); p.className=cls; p.textContent=txt; m.appendChild(p);
  }
}
setInterval(tick,500); tick();
</script></body></html>"""


def make_handler(game: LiveGame):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence access logs
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.startswith("/state"):
                self._send(json.dumps(game.snapshot()).encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

    return Handler


def serve_in_background(state, host: str = "127.0.0.1", port: int = 8000) -> str:
    """Serve ``state`` in a daemon thread; return the URL. Used to overlay a live
    web view on an already-running game (e.g. the MCP orchestrator)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://{host}:{port}"


def serve(config, host: str = "127.0.0.1", port: int = 8000, delay: float = 0.7):
    game = LiveGame(config, delay=delay)
    threading.Thread(target=game.run_forever, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(game))
    print(f"[live] watch the game at: http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[live] stopped.")
    finally:
        httpd.server_close()


def main():
    p = argparse.ArgumentParser(description="HW6 live web GUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--delay", type=float, default=0.7, help="seconds between half-moves")
    args = p.parse_args()
    serve(load_config(), args.host, args.port, args.delay)


if __name__ == "__main__":
    main()
