"""
mana.net.cli — the peer link from a command line.

Lives in the module rather than in a script, for the reason the exchange
command does: an installed MANA that cannot reach this is an installation
that cannot join anything, and a second copy in a script is how the two
come to disagree.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Sequence

from . import DEFAULT_PORT
from .link import LinkError, hello, serve, sync
from .peers import PeerBook

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


def _queue_path() -> Any:
    from ..cognition.exchange import default_queue_path
    return default_queue_path()


def _local_addresses(port: int) -> list:
    """Addresses a peer on the same network could reach us at.

    Printed with the key, because "add my key" is useless without
    somewhere to send it, and looking up your own LAN address is a step
    people get wrong.
    """
    import socket
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127.") and address not in found:
                found.append(address)
    except Exception:
        pass
    return [f"{a}:{port}" for a in found]


def command_line(argv: Sequence[str]) -> int:
    from ..core.identity import fingerprint, public_key

    book = PeerBook()
    action = (argv[0] if argv else "show").lower()

    if action in ("show", ""):
        print(f"этот экземпляр: {fingerprint()}")
        print(f"публичный ключ: {public_key().hex()}")
        addresses = _local_addresses(DEFAULT_PORT)
        if addresses:
            print(f"адреса в сети:  {', '.join(addresses)}")
        print()
        print("известные узлы:")
        print(book.describe())
        print()
        print("Спаривание взаимное: ваш ключ должен быть добавлен и там,")
        print("иначе они вас не пустят, а вы не пустите их.")
        return 0

    if action == "add":
        if len(argv) < 2:
            print("нужен ключ: --peer add <публичный ключ> [адрес] [имя]")
            return 2
        try:
            peer = book.add(argv[1], address=argv[2] if len(argv) > 2 else "",
                            label=argv[3] if len(argv) > 3 else "")
        except ValueError as exc:
            print(f"ключ не принят: {exc}")
            return 2
        print(f"добавлен {peer.fingerprint}"
              f"{' — ' + peer.label if peer.label else ''}"
              f"{' на ' + peer.address if peer.address else ''}")
        return 0

    if action == "remove":
        if len(argv) < 2:
            print("нужен отпечаток: --peer remove <отпечаток>")
            return 2
        print("удалён" if book.remove(argv[1]) else "такого узла нет")
        return 0

    if action == "serve":
        where = argv[1] if len(argv) > 1 else f"0.0.0.0:{DEFAULT_PORT}"
        host, _, port = where.rpartition(":")
        host = host or "0.0.0.0"
        try:
            port_number = int(port)
        except ValueError:
            print(f"не разобрать адрес {where!r}; нужно host:port")
            return 2
        httpd = serve(_queue_path(), host=host, port=port_number, book=book,
                      on_event=lambda text: print(f"  {text}"))
        print(f"узел слушает {host}:{port_number}")
        print(f"отпечаток: {fingerprint()}")
        print(f"ключ:      {public_key().hex()}")
        print()
        print("Отвечает только на обмен пакетами. Ни агента, ни песочницы,")
        print("ни доступа к 1С здесь нет — это отдельная точка входа.")
        print("Ctrl+C чтобы остановить.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            httpd.shutdown()
            print("остановлен")
        return 0

    if action == "hello":
        if len(argv) < 2:
            print("нужен адрес: --peer hello 192.168.0.5:8787")
            return 2
        try:
            greeting = hello(argv[1])
        except LinkError as exc:
            print(f"не дозвонился: {exc}")
            return 1
        print(f"отвечает {greeting.get('fingerprint')}")
        print(f"ключ:    {greeting.get('public_key')}")
        known = book.by_key(greeting.get("public_key", ""))
        print("в списке известных: " + ("да" if known else "НЕТ — добавьте "
                                        "командой --peer add"))
        return 0

    if action == "sync":
        if len(argv) < 2:
            print("нужен адрес: --peer sync 192.168.0.5:8787")
            return 2
        try:
            result = sync(argv[1], _queue_path(), book)
        except LinkError as exc:
            print(f"не вышло: {exc}")
            return 1
        print(f"обмен с {result['peer'] or '?'} ({result['address']})")
        print(f"  отправлено: {result['sent']['hypotheses']} гипотез, "
              f"{result['sent']['reports']} отчётов")
        print(f"  получено:   {result['received']['hypotheses']} гипотез, "
              f"{result['received']['reports']} отчётов"
              + (f", отклонено {result['received']['refused']}"
                 if result['received']['refused'] else ""))
        print()
        print("Чужие вердикты НЕ стали здесь истиной: гипотезы попали в")
        print("очередь, судить их будет местный эксперимент.")
        return 0

    print(f"неизвестная команда {action!r}; есть show, add, remove, "
          f"serve, hello, sync")
    return 2
