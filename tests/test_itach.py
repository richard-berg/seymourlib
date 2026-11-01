"""Tests for the iTach command-port helpers."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable

import pytest

from seymourlib import itach
from seymourlib.exceptions import SeymourConnectionError, SeymourProtocolError

Reply = Callable[[bytes], bytes | None]


class FakeCommandPort:
    """Answer iTach commands like an IP2SL's port 4998."""

    def __init__(self) -> None:
        self.baudrate = 9600
        self.commands: list[bytes] = []
        self.reply: Reply = self._settings_reply
        self.port = 0
        self._server: asyncio.Server | None = None

    def _settings_reply(self, command: bytes) -> bytes | None:
        if command.startswith(b"set_SERIAL,1:1,"):
            self.baudrate = int(command.split(b",")[2])
        return f"SERIAL,1:1,{self.baudrate},FLOW_NONE,PARITY_NO\r".encode()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            command = (await reader.readuntil(b"\r")).rstrip(b"\r")
        except asyncio.IncompleteReadError:
            return
        self.commands.append(command)
        if (reply := self.reply(command)) is None:
            writer.close()
            return
        writer.write(reply)
        await writer.drain()


@pytest.fixture
async def command_port() -> AsyncIterator[FakeCommandPort]:
    server = FakeCommandPort()
    await server.start()
    yield server
    await server.stop()


async def test_get_serial_settings(command_port: FakeCommandPort) -> None:
    settings = await itach.get_serial_settings("127.0.0.1", command_port.port)
    assert settings == itach.SerialSettings(9600, "FLOW_NONE", "PARITY_NO")
    assert not settings.suits_seymour
    assert command_port.commands == [b"get_SERIAL,1:1"]


async def test_set_serial_settings(command_port: FakeCommandPort) -> None:
    settings = await itach.set_serial_settings("127.0.0.1", port=command_port.port)
    assert settings == itach.SEYMOUR_SERIAL_SETTINGS
    assert settings.suits_seymour
    assert command_port.commands == [b"set_SERIAL,1:1,115200,FLOW_NONE,PARITY_NO"]


@pytest.mark.parametrize(
    "reply",
    [b"ERR_1:1,008\r", b"unknowncommand\r", b"SERIAL,1:2,9600,FLOW_NONE,PARITY_NO\r"],
)
async def test_unexpected_reply(command_port: FakeCommandPort, reply: bytes) -> None:
    command_port.reply = lambda _command: reply
    with pytest.raises(SeymourProtocolError):
        await itach.get_serial_settings("127.0.0.1", command_port.port)


async def test_bridge_hangs_up(command_port: FakeCommandPort) -> None:
    command_port.reply = lambda _command: None
    with pytest.raises(SeymourConnectionError):
        await itach.get_serial_settings("127.0.0.1", command_port.port)


async def test_no_bridge() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(SeymourConnectionError):
        await itach.get_serial_settings("127.0.0.1", port)


async def test_bridge_does_not_answer(command_port: FakeCommandPort) -> None:
    command_port.reply = lambda _command: b"SERIAL"  # never finished with \r
    with pytest.raises(SeymourConnectionError):
        await itach.get_serial_settings("127.0.0.1", command_port.port, command_timeout=0.1)
