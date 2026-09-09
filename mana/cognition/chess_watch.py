"""
mana.cognition.chess_watch — watching MANA play, and watching it think.

Why a window and not a log
---------------------------
A move list says what happened. The question this programme asks is why
the move that lost the game looked best at the time, and that is not in
the move list: it is in the scores the search assigned to the alternatives
it rejected, and in how far apart the top two were. Stage 0 measured 77%
of moves chosen with no margin at all -- ties broken by a shuffle. On a
board those are indistinguishable from decisions. Beside a panel showing
`e4 +0, d4 +0, Nf3 +0`, they are obvious at a glance.

So the page shows three things at once and they answer different questions:

    доска      что происходит
    трасса     что она рассматривала и с каким отрывом  — было ли это решением
    судья      сколько ход стоил                        — была ли это ошибка

Where the data comes from
--------------------------
Nowhere new. `chess_bot` already emits every position and every move
through `mana.events`, which is the bus the whole project reports itself
on; this subscribes to it and forwards anything carrying a `chess` payload
to the browser as server-sent events. No polling, no second source of
truth, and if this window is never opened the bot behaves identically.

Binding
-------
127.0.0.1, never 0.0.0.0, and the constructor refuses any other host. The
same rule `mana_desktop.server` follows: this serves an unauthenticated
page that reports what MANA is doing, and a bind to every interface would
put it on the network of whatever café the laptop is in.

No CDN, no board library
-------------------------
Pieces are Unicode on a CSS grid. A page that fetches a library renders a
blank square when the connection it needs is the one that is failing --
and the moment worth watching is often exactly that one.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .. import events

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The only address this may listen on.
LOOPBACK = "127.0.0.1"

DEFAULT_PORT = 8760

#: How many frames a newly opened window is caught up with. Enough to see
#: the current position and the last few moves' reasoning; not the whole
#: game, which would take longer to send than to watch.
REPLAY = 40


class _State:
    """The last frames, so a window opened mid-game is not blank."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.listeners: List["queue.Queue"] = []
        self.lock = threading.Lock()

    def push(self, frame: Dict[str, Any]) -> None:
        with self.lock:
            self.frames.append(frame)
            del self.frames[:-REPLAY]
            listeners = list(self.listeners)
        for sink in listeners:
            try:
                sink.put_nowait(frame)
            except Exception:
                pass                     # a window that stopped reading

    def subscribe(self) -> "queue.Queue":
        sink: "queue.Queue" = queue.Queue(maxsize=256)
        with self.lock:
            for frame in self.frames:
                try:
                    sink.put_nowait(frame)
                except Exception:
                    break
            self.listeners.append(sink)
        return sink

    def unsubscribe(self, sink: "queue.Queue") -> None:
        with self.lock:
            if sink in self.listeners:
                self.listeners.remove(sink)


STATE = _State()


def _sink(event: Any) -> None:
    payload = (getattr(event, "data", None) or {}).get("chess")
    if isinstance(payload, dict):
        STATE.push(payload)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass                              # the event bus is the log

    def do_GET(self) -> None:             # noqa: N802 (http.server API)
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/events":
            return self._stream()
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        sink = STATE.subscribe()
        try:
            while True:
                try:
                    frame = sink.get(timeout=15.0)
                    chunk = "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                except queue.Empty:
                    chunk = ": keep-alive\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError,
                ConnectionResetError, OSError):
            pass                          # window closed
        finally:
            STATE.unsubscribe(sink)


class _Server(ThreadingHTTPServer):
    """The window's own server, and the reason it refuses to share a port.

    `HTTPServer` sets `allow_reuse_address`, and on Windows that flag does
    not mean what it means elsewhere: a second process may bind a port
    another one is already listening on, both stay in LISTENING, and the
    first keeps taking the connections. Found exactly that way -- a second
    run of the watcher came up without an error and the browser went on
    showing the previous game, which is the worst possible failure for a
    window whose whole job is to show what is happening now.

    So: no reuse, and on Windows the exclusive flag that actually holds.
    A taken port is then an OSError the caller reports and plays without,
    instead of a window quietly showing something else.
    """
    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET,
                                       socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass                      # older Windows; the flag above still holds
        super().server_bind()


def start(port: int = DEFAULT_PORT, host: str = LOOPBACK) -> ThreadingHTTPServer:
    """Serve the window, and start forwarding events to it.

    Refuses any host but loopback rather than defaulting to it, so a
    caller cannot widen the binding by passing an argument that looks
    harmless.
    """
    if host != LOOPBACK:
        raise ValueError(f"наблюдение слушает только {LOOPBACK}, не {host}")
    events.subscribe(_sink)
    httpd = _Server((host, port), _Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="MANA-chess-watch",
                     daemon=True).start()
    return httpd


def url(httpd: ThreadingHTTPServer) -> str:
    return f"http://{LOOPBACK}:{httpd.server_address[1]}/"


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>MANA играет</title>
<style>
 :root{
   --ink:#1b1b1f; --dim:#6f7078; --line:#dcdce2; --ground:#f6f6f4;
   --panel:#ffffff; --light:#eeeed6; --dark:#7d955f; --mark:#c8532b;
   --good:#3f7a4a; --bad:#b23a2f;
 }
 @media (prefers-color-scheme: dark){:root{
   --ink:#e8e8ea; --dim:#9a9aa4; --line:#33343c; --ground:#17181c;
   --panel:#1f2026; --light:#b6b39a; --dark:#5c7345; --mark:#e07a4f;
   --good:#7bbf85; --bad:#e8756a;
 }}
 *{box-sizing:border-box}
 body{margin:0;background:var(--ground);color:var(--ink);
      font:14px/1.5 "Segoe UI",system-ui,sans-serif}
 header{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;
        padding:12px 18px;border-bottom:1px solid var(--line)}
 h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
 .dim{color:var(--dim)}
 main{display:grid;grid-template-columns:minmax(320px,440px) 1fr;
      gap:18px;padding:18px;align-items:start}
 @media (max-width:820px){main{grid-template-columns:1fr}}
 .panel{background:var(--panel);border:1px solid var(--line);
        border-radius:6px;padding:14px}
 .board{display:grid;grid-template-columns:repeat(8,1fr);
        aspect-ratio:1/1;border:1px solid var(--line);border-radius:4px;
        overflow:hidden;user-select:none}
 .sq{display:flex;align-items:center;justify-content:center;
     font-size:min(6vw,34px);line-height:1}
 .sq.l{background:var(--light)} .sq.d{background:var(--dark)}
 .sq.hit{box-shadow:inset 0 0 0 3px var(--mark)}
 .sq span{color:#111}
 .sq span.b{color:#111;text-shadow:0 0 1px #fff}
 /* Two lines, not two ends of one. Side by side they collided as soon
    as the summary grew: "полуход" and "26 ходов" ran together and the
    ply number wrapped away from its own label. */
 .meta{display:grid;gap:2px;margin-top:10px;
       font-size:13px;color:var(--dim)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th{text-align:left;font-weight:600;color:var(--dim);padding:4px 6px;
    border-bottom:1px solid var(--line);font-size:12px}
 td{padding:3px 6px;border-bottom:1px solid var(--line);
    font-variant-numeric:tabular-nums}
 td.n{text-align:right;width:5.5em}
 tr.pick td{font-weight:600}
 .bar{height:6px;background:var(--dark);border-radius:3px;opacity:.55}
 .tag{display:inline-block;padding:1px 7px;border-radius:10px;
      border:1px solid var(--line);font-size:12px}
 .tag.bad{color:var(--bad);border-color:var(--bad)}
 .tag.good{color:var(--good);border-color:var(--good)}
 .log{max-height:34vh;overflow:auto;margin-top:10px;font-size:13px}
 .log div{padding:2px 0;border-bottom:1px solid var(--line)}
 .k{color:var(--dim)}
 h2{font-size:13px;margin:0 0 8px;color:var(--dim);font-weight:600;
    letter-spacing:.06em;text-transform:uppercase}
</style></head><body>
<header>
  <h1>MANA играет</h1>
  <span class="dim" id="who">ожидание партии…</span>
  <a class="dim" id="link" href="#" target="_blank" rel="noopener"></a>
  <span class="dim" id="judge"></span>
  <span class="tag" id="ladder" hidden></span>
</header>
<main>
  <section class="panel">
    <div class="board" id="board"></div>
    <div class="meta"><span id="turn"></span><span id="score"></span></div>
  </section>
  <section class="panel">
    <h2>Ход размышлений</h2>
    <div id="why" class="dim">Первый ход ещё не сделан.</div>
    <table id="lines"></table>
    <h2 style="margin-top:16px">Партия</h2>
    <div class="log" id="log"></div>
  </section>
</main>
<script>
var GLYPH={P:"\\u2659",N:"\\u2658",B:"\\u2657",R:"\\u2656",Q:"\\u2655",K:"\\u2654",
           p:"\\u265F",n:"\\u265E",b:"\\u265D",r:"\\u265C",q:"\\u265B",k:"\\u265A"};
var flip=false, lastSquares=[];

function plural(n,one,few,many){
  var a=Math.abs(n)%100, b=a%10;
  if(a>10&&a<20) return many;
  if(b>1&&b<5) return few;
  if(b===1) return one;
  return many;
}

function draw(fen){
  var rows=fen.split(" ")[0].split("/");
  var cells=[];
  for(var r=0;r<8;r++){
    var line=[], parts=rows[r];
    for(var i=0;i<parts.length;i++){
      var c=parts[i];
      if(c>="1"&&c<="8"){for(var k=0;k<+c;k++) line.push("");}
      else line.push(c);
    }
    cells.push(line);
  }
  var order=[];
  for(var r=0;r<8;r++) for(var f=0;f<8;f++) order.push([r,f]);
  if(flip) order.reverse();
  var board=document.getElementById("board");
  board.innerHTML="";
  order.forEach(function(rf){
    var r=rf[0], f=rf[1], piece=cells[r][f];
    var name="abcdefgh"[f]+(8-r);
    var sq=document.createElement("div");
    sq.className="sq "+(((r+f)%2)?"d":"l")+(lastSquares.indexOf(name)>=0?" hit":"");
    if(piece){
      var s=document.createElement("span");
      s.textContent=GLYPH[piece];
      if(piece===piece.toLowerCase()) s.className="b";
      sq.appendChild(s);
    }
    board.appendChild(sq);
  });
}

function lines(t){
  var table=document.getElementById("lines");
  table.innerHTML="";
  if(!t||!t.considered||!t.considered.length) return;
  var head=table.insertRow();
  head.innerHTML="<th>ход</th><th>оценка</th><th></th>";
  var top=t.considered.slice(0,8);
  var hi=Math.max.apply(null,top.map(function(r){return Math.abs(r[1]);}))||1;
  top.forEach(function(row,i){
    var tr=table.insertRow();
    if(i===0) tr.className="pick";
    var w=Math.round(100*Math.abs(row[1])/hi);
    tr.innerHTML="<td>"+row[0]+"</td><td class='n'>"+
      (row[1]>0?"+":"")+row[1].toFixed(0)+"</td>"+
      "<td><div class='bar' style='width:"+w+"%'></div></td>";
  });
}

function note(f){
  var log=document.getElementById("log");
  var t=f.thought||{}, j=f.judged||{};
  var div=document.createElement("div");
  var bits=["<b>"+(f.last||"")+"</b>"];
  if(t.margin!==undefined)
    bits.push("<span class='k'>отрыв</span> "+
      (t.margin>0?"+":"")+t.margin.toFixed(0)+
      (t.close_call?" <span class='tag'>жребий</span>":""));
  if(j.loss!==undefined)
    bits.push("<span class='k'>потеря</span> "+j.loss.toFixed(0)+
      (j.blunder?" <span class='tag bad'>зевок</span>":
        j.mistake?" <span class='tag bad'>ошибка</span>":
          " <span class='tag good'>ок</span>"));
  div.innerHTML=bits.join(" &nbsp; ");
  log.insertBefore(div,log.firstChild);
  while(log.childNodes.length>120) log.removeChild(log.lastChild);
}

function apply(f){
  if(f.kind==="ready"){
    document.getElementById("who").textContent=
      (f.user?("аккаунт "+f.user+", ждём вызова"):"готова");
    if(f.judge) document.getElementById("judge").textContent=f.judge;
    return;
  }
  if(f.kind==="bench"){
    var L=f.ladder||{};
    var tag=document.getElementById("ladder");
    tag.hidden=false;
    tag.textContent="уровень "+L.level+" · подряд "+L.streak+"/"+L.needed+
      " · всего "+L.wins+" из "+L.games;
    return;
  }
  if(f.kind==="challenge"){
    document.getElementById("who").textContent=
      "вызов от "+f.by+": "+(f.accepted?"принят":f.reason);
    return;
  }
  if(f.fen){
    flip = (f.us==="black");
    lastSquares = f.last_uci ? [f.last_uci.slice(0,2), f.last_uci.slice(2,4)] : [];
    draw(f.fen);
    document.getElementById("who").textContent=
      "против "+(f.opponent||"?")+", MANA "+(f.us==="white"?"белыми":"чёрными");
    document.getElementById("turn").textContent=
      f.status!=="started" ? ("партия окончена: "+f.status+" "+(f.winner||""))
                           : ("ход "+(f.turn==="white"?"белых":"чёрных")+", полуход "+f.ply);
    var s=f.summary||{};
    document.getElementById("score").textContent=
      s.moves ? (s.moves+" "+plural(s.moves,"ход","хода","ходов")+
                 ", средняя потеря "+(s.mean_loss||0).toFixed(0)+
                 ", зевков "+s.blunders+
                 ", жребием "+Math.round(100*(s.close_share||0))+"%") : "";
    var link=document.getElementById("link");
    if(f.url){link.href=f.url; link.textContent=f.url;}
  }
  if(f.kind==="move"){
    var t=f.thought||{};
    lines(t);
    document.getElementById("why").innerHTML = t.chosen
      ? ("выбрала <b>"+t.chosen+"</b> из "+(t.considered||[]).length+" "+
         plural((t.considered||[]).length,"варианта","вариантов","вариантов")+
         ", глубина "+t.depth+", узлов "+t.nodes+
         (t.close_call?" — отрыва нет, решил жребий":" — отрыв "+t.margin.toFixed(0)))
      : "трасса выключена";
    note(f);
  }
}

var src=new EventSource("/events");
src.onmessage=function(e){ try{ apply(JSON.parse(e.data)); }catch(err){} };
draw("8/8/8/8/8/8/8/8 w - - 0 1");
</script></body></html>
"""
