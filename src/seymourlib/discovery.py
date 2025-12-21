"""Discovery helpers for Seymour transports."""

from __future__ import annotations

import asyncio
import socket
import struct
from dataclasses import dataclass
from typing import Any

from .transport import ITACH_SERIAL_PORT, SEYMOUR_BAUD_RATE, SerialTransport, TCPTransport

MCAST_GROUP = "239.255.250.250"
MCAST_PORT = 9131
MAX_MESSAGE_SIZE = 2048
DEFAULT_DISCOVERY_INTERVAL = 12.0  # in practice, beacons arrive every 10 seconds

__all__ = [
    "DiscoveryError",
    "SerialTransportCandidate",
    "TCPTransportCandidate",
    "enumerate_serial_transports",
    "enumerate_tcp_transports",
]


class DiscoveryError(RuntimeError):
    """Error raised when transport discovery fails."""


@dataclass(slots=True)
class TCPTransportCandidate:
    """Represents a discovered Global Caché IP2SL device."""

    transport: TCPTransport
    host: str
    port: int
    metadata: dict[str, str]
    raw_beacon: str


@dataclass(slots=True)
class SerialTransportCandidate:
    """Represents an available local serial port."""

    transport: SerialTransport
    device: str
    baudrate: int
    description: str
    hardware_id: str


async def enumerate_tcp_transports(
    interval: float = DEFAULT_DISCOVERY_INTERVAL,
    interface_ip: str | None = None,
) -> list[TCPTransportCandidate]:
    """Return TCP transports for all Global Caché devices seen via multicast."""

    if interval <= 0:
        raise ValueError("interval must be positive")

    try:
        sock = _create_multicast_socket(interface_ip)
    except OSError as exc:
        raise DiscoveryError(f"Unable to join discovery multicast group: {exc}") from exc

    loop = asyncio.get_running_loop()
    try:
        transport, protocol = await _bind_datagram_listener(sock)
    except Exception:
        sock.close()
        raise

    devices: dict[str, TCPTransportCandidate] = {}
    try:
        deadline = loop.time() + interval
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break

            try:
                record = await asyncio.wait_for(protocol.queue.get(), timeout=remaining)
            except TimeoutError:
                break

            if record is None:
                if protocol.error:
                    raise DiscoveryError(
                        f"Failed while reading discovery beacons: {protocol.error}"
                    ) from protocol.error
                break

            payload, (host, _port) = record

            raw_text = _safe_decode(payload)
            metadata: dict[str, str]
            try:
                metadata = _parse_amxb_payload(raw_text)
            except ValueError:
                metadata = {}

            devices[host] = TCPTransportCandidate(
                transport=TCPTransport(host, ITACH_SERIAL_PORT),
                host=host,
                port=ITACH_SERIAL_PORT,
                metadata=metadata,
                raw_beacon=raw_text,
            )
    finally:
        transport.close()

    return sorted(devices.values(), key=lambda candidate: candidate.host)


async def enumerate_serial_transports(
    baudrate: int = SEYMOUR_BAUD_RATE,
) -> list[SerialTransportCandidate]:
    """Return Serial transports for every available local serial port."""

    if baudrate <= 0:
        raise ValueError("baudrate must be positive")

    list_ports = _load_serial_list_ports()
    ports = list_ports.comports()

    candidates: list[SerialTransportCandidate] = []
    for port in ports:
        device = getattr(port, "device", None) or getattr(port, "name", None)
        if not device:
            continue

        description = getattr(port, "description", "") or ""
        hardware_id = getattr(port, "hwid", "") or ""

        candidates.append(
            SerialTransportCandidate(
                transport=SerialTransport(device, baudrate),
                device=device,
                baudrate=baudrate,
                description=description,
                hardware_id=hardware_id,
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.device)


def _safe_decode(payload: bytes) -> str:
    """Decode beacon payload to readable ASCII, preserving printable text."""

    return payload.decode("ascii", errors="replace").strip()


def _parse_amxb_payload(raw_text: str) -> dict[str, str]:
    """Parse AMXB Global Caché beacon text into a dictionary."""

    if not raw_text.startswith("AMXB"):
        raise ValueError("Not a Global Caché beacon")

    fields: dict[str, str] = {}
    parts = raw_text.split("<-")
    for part in parts[1:]:
        clean_part = part.rstrip("> \r\n\t")
        if not clean_part:
            continue

        if "=" in clean_part:
            key, value = clean_part.split("=", 1)
            fields[key] = value
        else:
            fields[clean_part] = ""

    if not fields:
        raise ValueError("Unable to parse beacon contents")

    return fields


def _create_multicast_socket(interface_ip: str | None) -> socket.socket:
    """Create and configure a multicast socket for Global Caché beacons."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))

    if interface_ip:
        interface_bytes = socket.inet_aton(interface_ip)
    else:
        interface_bytes = socket.inet_aton("0.0.0.0")

    mreq = struct.pack("=4s4s", socket.inet_aton(MCAST_GROUP), interface_bytes)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    return sock


class _DiscoveryDatagramProtocol(asyncio.DatagramProtocol):
    """Queue incoming datagrams so discovery can consume them with awaits."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[bytes, tuple[str, int]] | None] = asyncio.Queue()
        self.error: Exception | None = None

    def datagram_received(
        self, data: bytes, addr: tuple[str, int]
    ) -> None:  # pragma: no cover - trivial
        self.queue.put_nowait((data, addr))

    def error_received(self, exc: Exception) -> None:  # pragma: no cover - trivial
        self.error = exc
        self.queue.put_nowait(None)

    def connection_lost(self, exc: Exception | None) -> None:  # pragma: no cover - trivial
        if exc:
            self.error = exc
        self.queue.put_nowait(None)


async def _bind_datagram_listener(
    sock: socket.socket,
) -> tuple[asyncio.BaseTransport, _DiscoveryDatagramProtocol]:
    """Attach a queueing datagram protocol to the provided socket."""

    loop = asyncio.get_running_loop()
    protocol = _DiscoveryDatagramProtocol()
    transport, _ = await loop.create_datagram_endpoint(lambda: protocol, sock=sock)
    return transport, protocol


def _load_serial_list_ports() -> Any:  # pragma: no cover - thin wrapper
    """Return serial.tools.list_ports for environments with pyserial installed."""

    try:
        from serial.tools import list_ports
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise DiscoveryError(
            "pyserial is required to enumerate serial ports; install with 'pip install pyserial'"
        ) from exc

    return list_ports
