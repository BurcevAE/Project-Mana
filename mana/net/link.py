"""
mana.net.link — two installations exchanging bundles over a network.

Three routes and nothing else
------------------------------
    GET  /peer/hello    who is answering
    GET  /peer/bundle   give me what you have
    POST /peer/bundle   here is what I have

There is no fourth. This process holds no session, no agent and no
sandbox, and there is no route that could reach one -- which is the
property that makes it safe to listen on a network, and the reason it is
not simply another handler bolted onto `mana_desktop.server`.

Every request is signed
------------------------
The signature covers the method, the path, a digest of the body and a
timestamp, so a captured request cannot be replayed against a different
route or with a different body, and stops working once it ages past
CLOCK_SKEW_SECONDS. The key must already be in the peer book: signatures
prove who sent something, and the peer book decides whose word is worth
listening to at all.

Refusals are deliberately dull. "Не опознан" for every rejection --
unknown key, bad signature, stale timestamp -- because an error that
explains which of the three failed is an oracle for guessing the other
two.

What this does not do
----------------------
No discovery: an address has to be written down. Auto-discovery means
broadcasting on a network, which is a surface with no matching benefit
when the whole peer list is two machines somebody owns.

No encryption. Bundles carry no private data by construction, so what
this traffic needs is authenticity and integrity, and signatures give
both. That claim holds for exchange bundles and stops holding the moment
anything private travels this way.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import CLOCK_SKEW_SECONDS, DEFAULT_PORT
from .peers import PeerBook, UnknownPeer

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

HEADER_KEY = "X-Mana-Key"
HEADER_SIGNATURE = "X-Mana-Signature"
HEADER_TIME = "X-Mana-Time"

#: A bundle larger than this is refused unread. Without a cap, one POST
#: can spend all the memory this process has.
MAX_BODY = 8 * 1024 * 1024


class LinkError(RuntimeError):
    pass


def _signing_payload(method: str, path: str, body: bytes,
                     timestamp: str) -> Dict[str, Any]:
    """What the signature covers.

    The body as a digest rather than in full: the signature stays a fixed
    size, and a digest is as good as the bytes for proving they were not
    changed. Method and path are in there so a signature captured from a
    GET cannot be replayed onto the POST that accepts data.
    """
    return {"method": method.upper(), "path": path, "time": timestamp,
            "body": hashlib.blake2b(body or b"", digest_size=16).hexdigest()}


def sign_request(method: str, path: str, body: bytes = b"") -> Dict[str, str]:
    from ..core.identity import canonical, public_key, sign
    timestamp = f"{time.time():.0f}"
    payload = _signing_payload(method, path, body, timestamp)
    return {HEADER_KEY: public_key().hex(),
            HEADER_SIGNATURE: sign(canonical(payload)).hex(),
            HEADER_TIME: timestamp}


def check_request(headers: Any, method: str, path: str, body: bytes,
                  book: PeerBook) -> Any:
    """Return the peer that signed this, or raise. Never returns unsigned."""
    from ..core.identity import signature_holds

    key = headers.get(HEADER_KEY, "") or ""
    signature = headers.get(HEADER_SIGNATURE, "") or ""
    timestamp = headers.get(HEADER_TIME, "") or ""
    if not (key and signature and timestamp):
        raise UnknownPeer("запрос не подписан")

    try:
        drift = abs(time.time() - float(timestamp))
    except ValueError:
        raise UnknownPeer("метка времени нечитаема")
    if drift > CLOCK_SKEW_SECONDS:
        raise UnknownPeer("метка времени вне окна")

    peer = book.require(key)        # raises UnknownPeer for a key nobody added
    if not signature_holds(_signing_payload(method, path, body, timestamp),
                           key, signature):
        raise UnknownPeer("подпись не сходится")
    return peer


# ------------------------------------------------------------------- server


def make_handler(queue_path: Any, book: PeerBook, on_event=None):
    class PeerHandler(BaseHTTPRequestHandler):
        server_version = "MANA-peer"

        def log_message(self, *args: Any) -> None:
            """Quiet by default; the caller gets structured events instead
            of a second, differently-shaped log on stderr."""

        # ---------- plumbing ----------

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _refuse(self) -> None:
            # One message for every rejection. Saying which of key,
            # signature or clock failed would hand an attacker a way to
            # test each in turn.
            self._send(403, {"error": "не опознан"})

        def _body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return b""
            if length > MAX_BODY:
                raise LinkError("тело запроса слишком велико")
            return self.rfile.read(length) if length > 0 else b""

        def _peer(self, method: str, body: bytes):
            return check_request(self.headers, method, self.path, body, book)

        def _note(self, text: str) -> None:
            if on_event is not None:
                try:
                    on_event(text)
                except Exception:
                    pass

        # ---------- routes ----------

        def do_GET(self) -> None:                       # noqa: N802
            from ..cognition.exchange import Queue, export_bundle
            from ..core.identity import fingerprint, public_key

            if self.path == "/peer/hello":
                # Unsigned on purpose, and it reveals only what a peer
                # must know to add us: the public key is meant to be
                # published, and refusing to say it would make pairing
                # impossible without another channel.
                return self._send(200, {
                    "fingerprint": fingerprint(),
                    "public_key": public_key().hex(),
                    "product": "MANA"})

            if self.path != "/peer/bundle":
                return self._send(404, {"error": "нет такого адреса"})

            try:
                peer = self._peer("GET", b"")
            except UnknownPeer:
                return self._refuse()

            queue = Queue(queue_path)
            import tempfile
            with tempfile.TemporaryDirectory() as scratch:
                target = Path(scratch) / "bundle.json"
                export_bundle(target, queue.hypotheses(shareable_only=True),
                              queue.reports())
                payload = json.loads(target.read_text(encoding="utf-8"))
            book.seen(peer.fingerprint)
            self._note(f"отдан пакет узлу {peer.fingerprint}")
            self._send(200, payload)

        def do_POST(self) -> None:                      # noqa: N802
            from ..cognition.exchange import ExchangeError, Queue, import_bundle

            if self.path != "/peer/bundle":
                return self._send(404, {"error": "нет такого адреса"})
            try:
                body = self._body()
            except LinkError as exc:
                return self._send(413, {"error": str(exc)})
            try:
                peer = self._peer("POST", body)
            except UnknownPeer:
                return self._refuse()

            import tempfile
            queue = Queue(queue_path)
            with tempfile.TemporaryDirectory() as scratch:
                target = Path(scratch) / "bundle.json"
                target.write_bytes(body)
                try:
                    result = import_bundle(target, known=queue.known_ids())
                except ExchangeError as exc:
                    return self._send(400, {"error": str(exc)})
                absorbed = queue.absorb(result)

            book.seen(peer.fingerprint)
            self._note(f"принят пакет от {peer.fingerprint}: "
                       f"{absorbed['added']} гипотез")
            self._send(200, {"accepted_hypotheses": absorbed["added"],
                             "reports": absorbed["reports"],
                             "refused": len(result.refused)})

    return PeerHandler


def serve(queue_path: Any, host: str = "0.0.0.0", port: int = DEFAULT_PORT,
          book: Optional[PeerBook] = None, on_event=None) -> ThreadingHTTPServer:
    """Start listening. The caller decides the host, and must.

    0.0.0.0 is the default here and 127.0.0.1 is the default in
    `mana_desktop.server`, and the difference is the point: that one has
    no business being reachable, this one has no purpose unless it is.
    """
    book = book or PeerBook()
    handler = make_handler(queue_path, book, on_event)
    httpd = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, name="MANA-peer",
                     daemon=True).start()
    return httpd


# ------------------------------------------------------------------- client


def _call(address: str, method: str, path: str, body: bytes = b"",
          timeout: float = 30.0) -> Tuple[int, Any]:
    import urllib.error
    import urllib.request

    url = f"http://{address}{path}"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if path != "/peer/hello":
        headers.update(sign_request(method, path, body))
    request = urllib.request.Request(url, data=body or None, method=method,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": exc.reason}
    except Exception as exc:
        raise LinkError(f"{type(exc).__name__}: {exc}") from exc


def hello(address: str) -> Dict[str, Any]:
    """Ask an address who it is. Unsigned: this is how pairing starts."""
    status, payload = _call(address, "GET", "/peer/hello")
    if status != 200:
        raise LinkError(f"узел ответил {status}: {payload}")
    return payload


def sync(address: str, queue_path: Any,
         book: Optional[PeerBook] = None) -> Dict[str, Any]:
    """Push what we have, pull what they have. Both directions, one call.

    Push first: if the pull half fails, the other side still received
    ours, and a half-finished exchange that moved something is better
    than one that moved nothing.
    """
    from ..cognition.exchange import Queue, export_bundle, import_bundle

    book = book or PeerBook()
    queue = Queue(queue_path)

    # Who is actually answering at this address, and is it somebody this
    # installation recognises?
    #
    # Found by a test that expected a refusal and got a success. Signing
    # our requests proves us to THEM; it says nothing about them to us.
    # Without this check an impostor at the same address would receive our
    # bundle and serve back its own, with reports signed by a freshly
    # generated key -- which verify as signed, come from an instance
    # nobody added, and count as another environment in the replication
    # tally. That is the Sybil attack again, over a network this time.
    #
    # Bundles carry nothing secret, so what is at risk is not the data. It
    # is the count, which is the only thing the federation produces.
    greeting = hello(address)
    if book.by_key(greeting.get("public_key", "")) is None:
        raise LinkError(
            f"по адресу {address} отвечает узел, которого нет в списке "
            f"известных (отпечаток {greeting.get('fingerprint', '?')}). "
            f"Спаривание взаимное: добавьте его ключ командой "
            f"--peer add, если это ваша установка")

    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        outgoing = Path(scratch) / "out.json"
        export_bundle(outgoing, queue.hypotheses(shareable_only=True),
                      queue.reports())
        status, pushed = _call(address, "POST", "/peer/bundle",
                               outgoing.read_bytes())
        if status != 200:
            # The hint belongs here and not in the server's answer.
            # Pairing is mutual and that catches everybody once: adding
            # their key locally lets us call, and does nothing about
            # whether they will listen. The server stays deliberately
            # dull -- an error saying WHICH check failed is an oracle for
            # guessing the others -- while this side, which already knows
            # its own key is fine, may reasonably guess.
            hint = ""
            if status == 403:
                from ..core.identity import fingerprint
                hint = ""
            if status == 403:
                from ..core.identity import fingerprint
                hint = ("\nСкорее всего, на той стороне ещё не добавлен ваш "
                        "ключ. Спаривание взаимное: добавить их ключ у себя "
                        "позволяет позвонить и никак не влияет на то, станут "
                        "ли слушать.\nПусть выполнят там:  "
                        "MANA.exe --peer add <ваш ключ>\nВаш отпечаток: "
                        + fingerprint())
            raise LinkError(f"узел не принял пакет ({status}): "
                            f"{pushed.get('error', pushed)}{hint}")

        status, payload = _call(address, "GET", "/peer/bundle")
        if status != 200:
            raise LinkError(f"узел не отдал пакет ({status}): "
                            f"{payload.get('error', payload)}")
        incoming = Path(scratch) / "in.json"
        incoming.write_text(json.dumps(payload, ensure_ascii=False),
                            encoding="utf-8")
        result = import_bundle(incoming, known=queue.known_ids())
        absorbed = queue.absorb(result)

    if result.instance:
        book.seen(result.instance)
    return {
        "address": address,
        "sent": {"hypotheses": pushed.get("accepted_hypotheses", 0),
                 "reports": pushed.get("reports", 0)},
        "received": {"hypotheses": absorbed["added"],
                     "reports": absorbed["reports"],
                     "refused": len(result.refused)},
        "peer": result.instance,
    }
