"""Transport abstraction for Seymour controllers.

Every concrete transport is URL-based and opened with `serialx`, so any URL it
understands works: local devices (``/dev/ttyUSB0``, ``COM5``), raw TCP bridges
such as the Global Caché iTach IP2SL (``socket://host:4999``), RFC 2217 servers
and ESPHome serial proxies.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

import serialx

from .exceptions import SeymourTransportError

SEYMOUR_BAUD_RATE = 115200
ITACH_SERIAL_PORT = 4999
DEFAULT_CONNECT_TIMEOUT = 10.0


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
        try:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
        finally:
            self.writer = None
            self.reader = None

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

        Note: no timeout here; caller should use structured concurrency if desired.
        """
        if not self.reader:
            raise SeymourTransportError("Not connected")

        try:
            data = await asyncio.wait_for(self.reader.readuntil(b"]"), timeout=None)
            return data
        except asyncio.IncompleteReadError as exc:
            raise SeymourTransportError("Connection closed") from exc

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
