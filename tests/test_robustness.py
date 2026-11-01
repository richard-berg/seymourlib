"""Failure-mode tests against a fake IP2SL over real TCP (see conftest.py)."""

# mypy: disable-error-code="unreachable"

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from conftest import FakeIP2SL, reset

from seymourlib import protocol
from seymourlib import transport as transport_module
from seymourlib.client import SeymourClient
from seymourlib.exceptions import SeymourError, SeymourTransportError
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


class TestReceiveResyncs:
    @pytest.mark.parametrize(
        "noise",
        [b"\x00\xff junk", b"garbage]", b"\x00[0"],
        ids=["no-frame-end", "stray-frame-end", "torn-frame"],
    )
    async def test_noise_before_a_frame_is_discarded(self, ip2sl: FakeIP2SL, noise: bytes) -> None:
        async def noisy(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            writer.write(noise + srv.responses[frame])

        ip2sl.behavior = noisy
        client = make_client(ip2sl)
        status = await client.get_status()
        assert status.ratio is not None
        assert status.ratio.id == "178"
        await client.close()


class TestFrameTerminators:
    async def test_trailing_whitespace_is_not_a_warning(
        self, ip2sl: FakeIP2SL, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def crlf(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            writer.write(srv.responses[frame] + b"\r\n")

        caplog.set_level(logging.DEBUG, logger="seymourlib")
        ip2sl.behavior = crlf
        client = make_client(ip2sl)
        for _ in range(3):
            await client.get_status()
            await asyncio.sleep(0.02)
        await client.close()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "Drained 2 stale bytes" in caplog.text

    async def test_stale_frame_is_a_warning(
        self, ip2sl: FakeIP2SL, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def double(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            writer.write(srv.responses[frame] * 2)

        ip2sl.behavior = double
        client = make_client(ip2sl)
        await client.get_status()
        await asyncio.sleep(0.02)
        await client.get_status()
        await client.close()
        assert any(
            r.levelno == logging.WARNING and "Drained" in r.getMessage() for r in caplog.records
        )


class TestDiagnostics:
    async def test_response_containing_frame_ends_is_read_whole(self, ip2sl: FakeIP2SL) -> None:
        body = b'[01@D10{"ratios": [1, 2], "x": [3]}]'

        async def chunked(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            if frame.startswith(b"[01@"):
                for i in range(0, len(body), 8):
                    writer.write(body[i : i + 8])
                    await asyncio.sleep(0.01)
            else:
                writer.write(srv.responses[frame])

        ip2sl.behavior = chunked
        client = make_client(ip2sl, request_timeout=2)
        text = await client.get_diagnostics(protocol.DiagnosticOption.LIST_SETTINGS_JSON)
        assert text == body.decode()
        # Nothing was left behind to confuse the next request
        assert (await client.get_status()).ratio is not None
        await client.close()

    async def test_receive_until_idle_on_closed_connection(self, ip2sl: FakeIP2SL) -> None:
        async def hang_up(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            writer.close()

        ip2sl.behavior = hang_up
        transport = URLTransport(ip2sl.url, connect_timeout=1)
        await transport.connect()
        await transport.send(b"[01@D00]")
        with pytest.raises(SeymourTransportError):
            await transport.receive_until_idle(0.1)
        await transport.close()
        with pytest.raises(SeymourTransportError):
            await transport.receive_until_idle(0.1)

    async def test_receive_until_idle_stops_at_end_of_stream(self, ip2sl: FakeIP2SL) -> None:
        async def say_and_hang_up(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            writer.write(b"[01@x]")
            await writer.drain()
            writer.close()

        ip2sl.behavior = say_and_hang_up
        transport = URLTransport(ip2sl.url, connect_timeout=1)
        await transport.connect()
        await transport.send(b"[01@D00]")
        await asyncio.sleep(0.1)
        assert await transport.receive_until_idle(5) == b"[01@x]"
        await transport.close()
