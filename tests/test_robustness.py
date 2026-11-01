"""Failure-mode tests against a fake IP2SL over real TCP (see conftest.py)."""

# mypy: disable-error-code="unreachable"

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import FakeIP2SL, reset

from seymourlib import transport as transport_module
from seymourlib.client import SeymourClient
from seymourlib.exceptions import SeymourError
from seymourlib.transport import URLTransport


def make_client(ip2sl: FakeIP2SL, request_timeout: float = 0.3) -> SeymourClient:
    return SeymourClient(
        URLTransport(ip2sl.url, connect_timeout=1), request_timeout=request_timeout
    )


class TestCloseIsBestEffort:
    async def test_close_after_peer_reset_does_not_raise(self, ip2sl: FakeIP2SL) -> None:
        async def rst(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            reset(writer)

        ip2sl.behavior = rst
        client = make_client(ip2sl)
        await client.connect()
        with pytest.raises(SeymourError):
            await client.get_status()
        await client.close()

    async def test_close_is_bounded_when_peer_stops_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transport_module, "CLOSE_TIMEOUT", 0.2)
        peers: list[asyncio.StreamWriter] = []

        async def never_read(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            peers.append(writer)

        server = await asyncio.start_server(never_read, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        transport = URLTransport(f"socket://127.0.0.1:{port}", connect_timeout=1)
        await transport.connect()
        assert transport.writer is not None
        # More than the socket buffers hold, so bytes stay queued in the transport
        transport.writer.write(b"x" * 50_000_000)
        start = time.monotonic()
        await transport.close()
        assert time.monotonic() - start < 5
        assert transport.writer is None
        for peer in peers:
            peer.transport.abort()
        server.close()

    async def test_close_aborts_when_close_raises(self) -> None:
        class BrokenWriter:
            def __init__(self) -> None:
                self.transport = self
                self.aborted = False

            def close(self) -> None:
                raise RuntimeError("boom")

            def abort(self) -> None:
                self.aborted = True

        transport = URLTransport("socket://127.0.0.1:1")
        writer = BrokenWriter()
        transport.writer = writer  # type: ignore[assignment]
        await transport.close()
        assert writer.aborted
        assert transport.writer is None
