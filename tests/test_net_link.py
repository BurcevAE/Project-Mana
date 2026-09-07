"""
tests/test_net_link.py — the link between two installations.

The property under test is narrow and load-bearing: this endpoint is
reachable from a network, and `mana_desktop.server` must never be. That
one answers /api/ask, which runs code in the sandbox, and
/api/onec/confirm, which writes to a live trade database. This one moves
exchange bundles and can reach neither.

Two of these tests exist because a run caught a real hole: `sync`
succeeded against a peer the caller had not added, because signing our
requests proves us to them and says nothing about them to us.
"""
from __future__ import annotations

import time

import pytest

from mana.net import CLOCK_SKEW_SECONDS, link, peers


@pytest.fixture
def book(tmp_path, monkeypatch):
    from mana.core import identity
    monkeypatch.setenv(identity.INSTANCE_ENV, "link-test")
    identity.reset_cache()
    return peers.PeerBook(tmp_path / "peers.json")


@pytest.fixture
def my_key():
    from mana.core import identity
    return identity.public_key().hex()


class Headers(dict):
    """Just enough of an http.client message for check_request."""


# ------------------------------------------------------------- the peer book

def test_the_fingerprint_is_derived_not_accepted(book, my_key):
    """Taking a claimed fingerprint would let somebody file their key
    under another instance's name -- the impersonation the signature check
    catches one level down, arriving through the front door instead."""
    from mana.core import identity
    peer = book.add(my_key, label="сам себе")
    assert peer.fingerprint == identity.fingerprint(my_key)


def test_a_key_of_the_wrong_length_is_refused(book):
    with pytest.raises(ValueError) as caught:
        book.add("aabbcc")
    assert "32" in str(caught.value)


def test_a_key_that_is_not_hex_is_refused(book):
    with pytest.raises(ValueError):
        book.add("это не ключ")


def test_lookup_requires_the_exact_key_not_just_the_digest(book, my_key):
    """The stored key is compared, not only its 8-character digest.

    Matching on the digest alone would make a hash collision a credential,
    and turn an identifier into an authenticator.
    """
    peer = book.add(my_key)
    record = book._load()[peer.fingerprint]
    record["public_key"] = "00" * 32
    book._save({peer.fingerprint: record})
    assert book.by_key(my_key) is None


def test_an_unknown_key_raises_rather_than_returning_none(book, my_key):
    with pytest.raises(peers.UnknownPeer):
        book.require(my_key)


# ------------------------------------------------------ the signed request

def _headers(method, path, body=b""):
    return Headers(link.sign_request(method, path, body))


def test_a_signed_request_from_a_known_peer_is_accepted(book, my_key):
    book.add(my_key)
    headers = _headers("GET", "/peer/bundle")
    assert link.check_request(headers, "GET", "/peer/bundle", b"", book)


def test_an_unsigned_request_is_refused(book):
    with pytest.raises(peers.UnknownPeer):
        link.check_request(Headers(), "GET", "/peer/bundle", b"", book)


def test_a_signature_for_one_route_does_not_work_on_another(book, my_key):
    """The method and path are inside the signed payload, so a request
    captured from the read endpoint cannot be replayed onto the one that
    accepts data."""
    book.add(my_key)
    headers = _headers("GET", "/peer/bundle")
    with pytest.raises(peers.UnknownPeer):
        link.check_request(headers, "POST", "/peer/bundle", b"", book)


def test_a_signature_does_not_carry_over_to_a_different_body(book, my_key):
    book.add(my_key)
    headers = _headers("POST", "/peer/bundle", b'{"a":1}')
    with pytest.raises(peers.UnknownPeer):
        link.check_request(headers, "POST", "/peer/bundle", b'{"a":2}', book)


def test_a_stale_request_is_refused(book, my_key):
    """Replay protection: a captured request stops working once it ages
    out, so a recording is not a key."""
    book.add(my_key)
    headers = _headers("GET", "/peer/bundle")
    headers[link.HEADER_TIME] = f"{time.time() - CLOCK_SKEW_SECONDS - 60:.0f}"
    with pytest.raises(peers.UnknownPeer):
        link.check_request(headers, "GET", "/peer/bundle", b"", book)


def test_a_request_from_a_key_nobody_added_is_refused(book, my_key):
    headers = _headers("GET", "/peer/bundle")
    with pytest.raises(peers.UnknownPeer):
        link.check_request(headers, "GET", "/peer/bundle", b"", book)


# ------------------------------------------------------- shape of the server

def test_the_peer_server_has_three_routes_and_no_more():
    """It listens on a network, so its surface has to be readable in one
    sitting. A fourth route is a decision, not a detail."""
    import inspect
    source = inspect.getsource(link.make_handler)
    assert source.count('self.path == "/peer/hello"') == 1
    assert source.count('self.path != "/peer/bundle"') == 2   # GET and POST
    for forbidden in ("/api/ask", "session", "sandbox", "agent"):
        assert forbidden not in source


def test_every_refusal_says_the_same_thing():
    """An error that explains WHICH of key, signature or clock failed is
    an oracle for guessing the other two."""
    import inspect
    source = inspect.getsource(link.make_handler)
    assert source.count('"не опознан"') == 1
    assert "_refuse()" in source


def test_the_body_size_is_capped():
    """Without a cap one POST spends all the memory this process has."""
    assert link.MAX_BODY <= 32 * 1024 * 1024
    import inspect
    assert "MAX_BODY" in inspect.getsource(link.make_handler)


def test_sync_refuses_an_address_answering_with_an_unknown_key(
        book, monkeypatch, tmp_path):
    """The hole a run caught: signing our requests proves us to them and
    says nothing about them to us.

    Without this an impostor at the same address takes our bundle and
    serves back its own, with reports signed by a freshly generated key --
    which verify as signed, come from an instance nobody added, and count
    as another environment in the replication tally.
    """
    monkeypatch.setattr(link, "hello", lambda address: {
        "fingerprint": "deadbeef", "public_key": "11" * 32})
    with pytest.raises(link.LinkError) as caught:
        link.sync("127.0.0.1:9", tmp_path / "q.json", book)
    assert "известных" in str(caught.value)
