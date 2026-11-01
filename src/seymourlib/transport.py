"""Transport abstraction for Seymour controllers.

Every concrete transport is URL-based and opened with `serialx`, so any URL it
understands works: local devices (``/dev/ttyUSB0``, ``COM5``), raw TCP bridges
such as the Global Caché iTach IP2SL (``socket://host:4999``), RFC 2217 servers
and ESPHome serial proxies.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod

import serialx

from .exceptions import SeymourTransportError

_LOGGER = logging.getLogger(__name__)

SEYMOUR_BAUD_RATE = 115200
ITACH_SERIAL_PORT = 4999
DEFAULT_CONNECT_TIMEOUT = 10.0
CLOSE_TIMEOUT = 2.0


def tcp_url(host: str, port: int = ITACH_SERIAL_PORT) -> str:
    """Return the ``socket://`` URL for a raw TCP serial bridge such as the IP2SL."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"socket://{host}:{port}"


class SeymourTransport:
    """Abstract transport used by the SeymourClient."""

    def __init__(self) -> None:
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        """Close the connection; never raises, and never takes long.

        Closing is cleanup, so it must not fail just because the connection
        already broke: ``wait_closed()`` re-raises the error that ended the
        connection, such as a reset by the IP2SL.  A peer that stops reading
        could also stall the close while unsent bytes drain, so the wait is
        bounded and the connection aborted when it runs out.
        """
        writer, self.writer, self.reader = self.writer, None, None
        if writer is None:
            return
        try:
            writer.close()
            async with asyncio.timeout(CLOSE_TIMEOUT):
                await writer.wait_closed()
        except TimeoutError:
            _LOGGER.debug("Timed out closing the connection; aborting it")
            writer.transport.abort()
        except Exception as exc:  # noqa: BLE001 - closing is best-effort
            _LOGGER.debug("Ignoring error while closing the connection: %r", exc)
            writer.transport.abort()

    async def send(self, data: bytes) -> None:
        """Send one data frame to transport, blocking until the complete frame is sent."""
        assert data.startswith(b"["), "Frame must start with '['"
        assert data.endswith(b"]"), "Frame must end with ']'"

        if not self.writer:
            raise SeymourTransportError("Not connected")

        self.writer.write(data)
        await self.writer.drain()

    async def receive(self) -> bytes:
        """
        Get one data frame from transport, blocking until frame end (']') is received.

        Bytes before the frame's '[' are line noise, such as the garbage a
        UART emits when the controller powers up, and are discarded.

        Note: no timeout here; caller should use structured concurrency if desired.
        """
        if not self.reader:
            raise SeymourTransportError("Not connected")

        while True:
            try:
                data = await self.reader.readuntil(b"]")
            except asyncio.IncompleteReadError as exc:
                raise SeymourTransportError("Connection closed") from exc
            start = data.rfind(b"[")
            if start < 0:
                _LOGGER.debug("Discarding bytes outside a frame: %r", data)
                continue
            if start > 0:
                _LOGGER.debug("Discarding bytes before a frame: %r", data[:start])
            return data[start:]

    async def receive_until_idle(self, idle: float) -> bytes:
        """Get a response of unknown length: every byte until the line goes quiet.

        Some responses, such as the JSON diagnostics, contain ']' themselves,
        so they can't be read up to the frame end.  This waits for the first
        byte without a timeout, like receive(), and then for ``idle`` seconds
        of silence.
        """
        if not self.reader:
            raise SeymourTransportError("Not connected")

        data = await self.reader.read(4096)
        if not data:
            raise SeymourTransportError("Connection closed")
        while True:
            try:
                async with asyncio.timeout(idle):
                    chunk = await self.reader.read(4096)
            except TimeoutError:
                return data
            if not chunk:
                return data
            data += chunk

    async def drain_read_buffer(self) -> bytes:
        """Discard any stale bytes sitting in the receive buffer.

        Returns whatever was drained (may be empty).  This is called
        before each send to prevent a desynchronised buffer from causing
        the next ``receive()`` to return a frame belonging to a previous
        request.
        """
        if not self.reader:
            return b""

        drained = b""
        # StreamReader buffers incoming data internally.  A non-blocking
        # read of whatever is already buffered is safe; we set a tiny
        # timeout so that we never actually block waiting for the device.
        while True:
            try:
                chunk = await asyncio.wait_for(self.reader.read(4096), timeout=0.01)
                if not chunk:
                    break
                drained += chunk
            except TimeoutError:
                break
        return drained


class URLTransport(SeymourTransport):
    """Transport for any serial URL supported by serialx."""

    def __init__(
        self,
        url: str,
        baudrate: int = SEYMOUR_BAUD_RATE,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        super().__init__()
        self.url = url
        self.baudrate = baudrate
        self.connect_timeout = connect_timeout

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={serialx.redact_uri(self.url)!r})"

    async def connect(self) -> None:
        # Close any existing connection first.  The IP2SL only accepts one
        # TCP connection per serial port, so a leaked socket would hold that
        # slot and block every future connect.
        await self.close()
        # A busy IP2SL slot can accept the TCP handshake and then never
        # respond, so bound the whole open with a timeout.
        async with asyncio.timeout(self.connect_timeout):
            self.reader, self.writer = await serialx.open_serial_connection(
                url=self.url, baudrate=self.baudrate
            )


class TCPTransport(URLTransport):
    """Transport for a raw TCP serial bridge such as the Global Caché iTach IP2SL."""

    def __init__(
        self,
        host: str,
        port: int = ITACH_SERIAL_PORT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        super().__init__(tcp_url(host, port), connect_timeout=connect_timeout)
        self.host = host
        self.port = port


class SerialTransport(URLTransport):
    """Transport for a local serial device, e.g. ``/dev/ttyUSB0`` or ``COM5``."""

    def __init__(
        self,
        port: str,
        baudrate: int = SEYMOUR_BAUD_RATE,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        super().__init__(port, baudrate, connect_timeout)
        self.port = port
