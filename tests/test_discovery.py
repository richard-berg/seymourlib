"""Tests for the discovery helper APIs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import serialx

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
    assert candidates[0].url == f"socket://192.168.1.7:{ITACH_SERIAL_PORT}"
    assert transport.closed


def _port_info(device: str, **kwargs: object) -> serialx.SerialPortInfo:
    fields: dict[str, object] = {
        "device": device,
        "resolved_device": device,
        "vid": None,
        "pid": None,
        "serial_number": None,
        "manufacturer": None,
        "product": None,
        "bcd_device": None,
        "interface_description": None,
        "interface_num": None,
        "port_number": None,
    }
    fields.update(kwargs)
    return serialx.SerialPortInfo(**fields)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_enumerate_serial_transports_uses_listed_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        _port_info(
            "/dev/ttyUSB0",
            vid=0x0403,
            pid=0x6001,
            serial_number="AH05TTD9",
            product="FT232R USB UART",
        ),
        _port_info("COM1", interface_description="Communications Port"),
        _port_info(""),
    ]

    async def _fake_list(**_kwargs: object) -> list[serialx.SerialPortInfo]:
        return ports

    monkeypatch.setattr(serialx, "async_list_serial_ports", _fake_list)

    candidates = await enumerate_serial_transports(baudrate=9600)

    assert [candidate.device for candidate in candidates] == ["/dev/ttyUSB0", "COM1"]
    lookup = {candidate.device: candidate for candidate in candidates}
    assert lookup["COM1"].baudrate == 9600
    assert lookup["COM1"].transport.port == "COM1"
    assert lookup["COM1"].url == "COM1"
    assert lookup["COM1"].description == "Communications Port"
    assert lookup["COM1"].hardware_id == ""
    usb = lookup["/dev/ttyUSB0"]
    assert usb.transport.baudrate == 9600
    assert usb.description == "FT232R USB UART"
    assert usb.hardware_id == "USB VID:PID=0403:6001 SER=AH05TTD9"


@pytest.mark.asyncio
async def test_enumerate_serial_transports_hardware_id_without_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_list(**_kwargs: object) -> list[serialx.SerialPortInfo]:
        return [_port_info("/dev/ttyACM0", vid=0x2341, pid=0x0043)]

    monkeypatch.setattr(serialx, "async_list_serial_ports", _fake_list)

    (candidate,) = await enumerate_serial_transports()

    assert candidate.hardware_id == "USB VID:PID=2341:0043"
    assert candidate.baudrate == SEYMOUR_BAUD_RATE


@pytest.mark.asyncio
async def test_enumerate_serial_transports_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise(**_kwargs: object) -> list[serialx.SerialPortInfo]:
        raise PermissionError("denied")

    monkeypatch.setattr(serialx, "async_list_serial_ports", _raise)

    with pytest.raises(DiscoveryError, match="denied"):
        await enumerate_serial_transports(SEYMOUR_BAUD_RATE)


@pytest.mark.asyncio
async def test_enumerate_serial_transports_rejects_bad_baudrate() -> None:
    with pytest.raises(ValueError):
        await enumerate_serial_transports(0)
