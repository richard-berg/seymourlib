"""Shared fixtures: a real TCP server that stands in for a Global Caché IP2SL.

Unlike the mock transports, this exercises the real serialx ``socket://``
stack, so tests see the same resets, EOFs, and buffering as production.
"""

from __future__ import annotations

import asyncio
import socket
import struct
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

STATUS_RESP = b"[01P178]"
POSITIONS_RESP = b"[012T50.0B25.0]"
SYSTEM_INFO_RESP = b"[01PRH-123             0123.00069.1SS-0325-Berg TB]"

DEFAULT_RESPONSES = {
    b"[01S]": STATUS_RESP,
    b"[01P]": POSITIONS_RESP,
    b"[01Y]": SYSTEM_INFO_RESP,
}

Behavior = Callable[["FakeIP2SL", int, bytes, asyncio.StreamWriter], Awaitable[None]]


async def answer(srv: FakeIP2SL, conn: int, frame: bytes, writer: asyncio.StreamWriter) -> None:
    """Behave like a healthy controller: answer queries, ignore commands."""
    if (response := srv.responses.get(frame)) is not None:
        writer.write(response)
        await writer.drain()


def reset(writer: asyncio.StreamWriter) -> None:
    """Abort the connection with a TCP RST."""
    sock = writer.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


class FakeIP2SL:
    """A scriptable TCP server; ``behavior`` decides how to treat each frame."""

    def __init__(self, behavior: Behavior = answer) -> None:
        self.behavior = behavior
        self.responses = dict(DEFAULT_RESPONSES)
        self.connections = 0
        self.active = 0
        self.frames: list[bytes] = []
        self.writers: list[asyncio.StreamWriter] = []
        self.port = 0
        self._server: asyncio.Server | None = None

    @property
    def url(self) -> str:
        return f"socket://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        for writer in self.writers:
            writer.transport.abort()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def count(self, frame: bytes) -> int:
        return self.frames.count(frame)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.active += 1
        self.writers.append(writer)
        conn = self.connections
        try:
            while True:
                frame = await reader.readuntil(b"]")
                self.frames.append(frame)
                await self.behavior(self, conn, frame, writer)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            pass
        finally:
            self.active -= 1


@pytest.fixture
async def ip2sl() -> AsyncIterator[FakeIP2SL]:
    """Return a running fake IP2SL; set ``behavior`` to make it misbehave."""
    server = FakeIP2SL()
    await server.start()
    yield server
    await server.stop()
