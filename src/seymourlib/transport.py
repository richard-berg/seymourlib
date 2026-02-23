"""Transport abstraction: TCPTransport (for IP2SL) + SerialTransport (local serial)."""

from __future__ import annotations

import asyncio
from abc import abstractmethod

from .exceptions import SeymourTransportError

SEYMOUR_BAUD_RATE = 115200
ITACH_SERIAL_PORT = 4999


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


class TCPTransport(SeymourTransport):
    def __init__(self, host: str, port: int = ITACH_SERIAL_PORT) -> None:
        super().__init__()
        self.host = host
        self.port = port

    async def connect(self) -> None:
        # Close any existing connection first to avoid leaking the TCP
        # socket.  The IP2SL only accepts one connection per serial port;
        # a leaked socket holds that slot and blocks all future connects.
        await self.close()
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)


class SerialTransport(SeymourTransport):
    def __init__(self, port: str, baudrate: int = SEYMOUR_BAUD_RATE) -> None:
        super().__init__()
        self.port = port
        self.baudrate = baudrate

    async def connect(self) -> None:
        await self.close()
        try:
            import serial_asyncio
        except Exception as exc:
            raise SeymourTransportError("serial_asyncio not installed") from exc

        self.reader, self.writer = await serial_asyncio.open_serial_connection(
            url=self.port, baudrate=self.baudrate
        )
