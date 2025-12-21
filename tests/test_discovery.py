"""Tests for the discovery helper APIs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from seymourlib import discovery
from seymourlib.discovery import (
    DiscoveryError,
    _parse_amxb_payload,
    enumerate_serial_transports,
    enumerate_tcp_transports,
)
from seymourlib.transport import ITACH_SERIAL_PORT, SEYMOUR_BAUD_RATE


def test_parse_amxb_payload_parses_expected_fields() -> None:
    payload = (
        "AMXB<-UUID=GlobalCache_000C1E060E8E><-Make=GlobalCache><-Model=iTachIP2SL>"
        "<-Status=Ready><-Config-URL=http://192.168.1.7>"
    )

    parsed = _parse_amxb_payload(payload)

    assert parsed["UUID"] == "GlobalCache_000C1E060E8E"
    assert parsed["Model"] == "iTachIP2SL"
    assert parsed["Status"] == "Ready"
    assert parsed["Config-URL"] == "http://192.168.1.7"


@pytest.mark.asyncio
async def test_enumerate_tcp_transports_returns_unique_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        (
            b"AMXB<-UUID=One><-Model=iTachIP2SL><-Status=Ready>\r\n",
            ("192.168.1.7", discovery.MCAST_PORT),
        ),
        (
            b"AMXB<-UUID=Two><-Model=iTachIP2SL><-Status=Ready>\r\n",
            ("192.168.1.8", discovery.MCAST_PORT),
        ),
    ]

    class DummyTransport:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    queue: asyncio.Queue[tuple[bytes, tuple[str, int]] | None] = asyncio.Queue()
    for packet in responses:
        queue.put_nowait(packet)
    queue.put_nowait(None)

    protocol = SimpleNamespace(queue=queue, error=None)
    transport = DummyTransport()

    monkeypatch.setattr(discovery, "_create_multicast_socket", lambda _ip: object())

    async def _fake_bind(sock: object) -> tuple[DummyTransport, SimpleNamespace]:
        assert sock is not None
        return transport, protocol

    monkeypatch.setattr(discovery, "_bind_datagram_listener", _fake_bind)

    candidates = await enumerate_tcp_transports(interval=0.5)

    assert len(candidates) == 2
    assert {candidate.host for candidate in candidates} == {"192.168.1.7", "192.168.1.8"}
    assert all(candidate.port == ITACH_SERIAL_PORT for candidate in candidates)
    assert transport.closed


@pytest.mark.asyncio
async def test_enumerate_serial_transports_uses_loaded_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        SimpleNamespace(device="COM1", description="USB Serial", hwid="ABC"),
        SimpleNamespace(device="/dev/ttyUSB0", description="USB", hwid="XYZ"),
    ]

    class DummyListPorts:
        def comports(self) -> list[SimpleNamespace]:
            return ports

    monkeypatch.setattr(discovery, "_load_serial_list_ports", lambda: DummyListPorts())

    candidates = await enumerate_serial_transports(baudrate=9600)

    assert len(candidates) == 2
    lookup = {candidate.device: candidate for candidate in candidates}
    assert lookup["COM1"].baudrate == 9600
    assert lookup["COM1"].transport.port == "COM1"
    assert lookup["/dev/ttyUSB0"].transport.port == "/dev/ttyUSB0"
    assert all(candidate.hardware_id for candidate in lookup.values())


@pytest.mark.asyncio
async def test_enumerate_serial_transports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise DiscoveryError("missing pyserial")

    monkeypatch.setattr(discovery, "_load_serial_list_ports", _raise)

    with pytest.raises(DiscoveryError, match="missing pyserial"):
        await enumerate_serial_transports(SEYMOUR_BAUD_RATE)
