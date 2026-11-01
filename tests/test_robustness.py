"""Failure-mode tests against a fake IP2SL over real TCP (see conftest.py)."""

# mypy: disable-error-code="unreachable"

from __future__ import annotations

import asyncio
import logging
import socket
import time

import pytest
from conftest import FakeIP2SL, answer, reset

from seymourlib import protocol
from seymourlib import transport as transport_module
from seymourlib.client import SeymourClient
from seymourlib.exceptions import (
    SeymourClientClosedError,
    SeymourConnectionError,
    SeymourError,
    SeymourTransportError,
)
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


class TestLogging:
    async def test_frames_are_logged_to_the_library_logger(
        self, ip2sl: FakeIP2SL, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        client = make_client(ip2sl)
        await client.get_status()
        await client.close()
        frame_records = [r for r in caplog.records if "frame" in r.getMessage()]
        assert frame_records
        assert {r.name for r in frame_records} == {"seymourlib.client"}

    async def test_time_since_last_success_ignores_wall_clock_jumps(
        self, ip2sl: FakeIP2SL, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = make_client(ip2sl)
        await client.get_status()
        # NTP steps the wall clock back an hour
        monkeypatch.setattr(time, "time", lambda: client._last_successful_operation - 3600)
        assert 0 <= client.connection_stats["time_since_last_success"] < 5
        await client.close()


class TestCloseIsFinal:
    async def test_request_in_flight_does_not_reconnect_after_close(self, ip2sl: FakeIP2SL) -> None:
        async def silent_first(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            if conn > 1:
                await answer(srv, conn, frame, writer)

        ip2sl.behavior = silent_first
        client = make_client(ip2sl)
        await client.connect()
        request = asyncio.create_task(client.get_status())
        await asyncio.sleep(0.4)  # the first attempt timed out; waiting to retry
        await client.close()
        with pytest.raises(SeymourClientClosedError):
            await request
        await asyncio.sleep(0.1)
        assert not client.is_connected
        assert ip2sl.active == 0
        assert ip2sl.connections == 1

    async def test_closed_client_refuses_requests(self, ip2sl: FakeIP2SL) -> None:
        client = make_client(ip2sl)
        await client.close()
        with pytest.raises(SeymourClientClosedError):
            await client.connect()
        with pytest.raises(SeymourClientClosedError):
            await client.get_status()
        assert ip2sl.connections == 0

    async def test_disconnect_keeps_the_client_usable(self, ip2sl: FakeIP2SL) -> None:
        client = make_client(ip2sl)
        await client.get_status()
        await client.disconnect()
        assert not client.is_connected
        await client.get_status()
        assert ip2sl.connections == 2
        await asyncio.sleep(0.05)
        assert ip2sl.active == 1
        await client.close()


class TestRetriesReconnect:
    @pytest.mark.parametrize("how", ["fin", "rst"])
    async def test_request_survives_a_dropped_connection(self, ip2sl: FakeIP2SL, how: str) -> None:
        async def drop_first(
            srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter
        ) -> None:
            if conn > 1:
                await answer(srv, conn, frame, writer)
            elif how == "fin":
                writer.close()
            else:
                reset(writer)

        ip2sl.behavior = drop_first
        client = make_client(ip2sl)
        await client.connect()
        status = await client.get_status()
        assert status.ratio is not None
        assert ip2sl.connections == 2
        await client.close()

    async def test_lock_is_free_while_waiting_to_retry(self) -> None:
        # Nothing listens on this port, so every connect is refused at once
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = SeymourClient(
            URLTransport(f"socket://127.0.0.1:{port}", connect_timeout=1), request_timeout=0.3
        )
        request = asyncio.create_task(client.get_status())
        await asyncio.sleep(0.1)  # the first connect failed; now waiting to retry
        await asyncio.wait_for(client.disconnect(), timeout=0.2)
        with pytest.raises(SeymourConnectionError):
            await request
        await client.close()


class TestUnexpectedErrors:
    async def test_unexpected_error_resets_the_connection(
        self, ip2sl: FakeIP2SL, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = make_client(ip2sl)
        await client.get_status()
        real_send = client.transport.send
        calls = 0

        async def broken_send(data: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AssertionError("transport in a bad state")
            await real_send(data)

        monkeypatch.setattr(client.transport, "send", broken_send)
        with pytest.raises(SeymourConnectionError) as exc_info:
            await client.get_status()
        assert isinstance(exc_info.value.__cause__, AssertionError)
        assert not client.is_connected
        # The next request starts over on a new connection
        assert (await client.get_status()).ratio is not None
        assert ip2sl.connections == 2
        await client.close()
