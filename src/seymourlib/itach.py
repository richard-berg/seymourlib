"""Settings of a Global Caché iTach bridge, through its command port.

The IP2SL forwards bytes to the controller at whatever serial settings it was
configured with; a factory reset sets 9600 baud, where the controller, at
115200, can't understand it.  To the client that only looks like a silent
controller, so these helpers let applications check, and fix, the settings.
"""

from __future__ import annotations

import asyncio
import re

import attrs

from .exceptions import SeymourConnectionError, SeymourProtocolError
from .transport import SEYMOUR_BAUD_RATE

ITACH_COMMAND_PORT = 4998
# The serial port of an IP2SL is port 1 of module 1, served at TCP port 4999
SERIAL_CONNECTOR = "1:1"
FLOW_NONE = "FLOW_NONE"
PARITY_NO = "PARITY_NO"
DEFAULT_TIMEOUT = 5.0

_SERIAL_RE = re.compile(
    r"SERIAL,(?P<connector>\d+:\d+),(?P<baudrate>\d+),(?P<flow>FLOW_\w+),(?P<parity>PARITY_\w+)"
)


@attrs.frozen
class SerialSettings:
    """Serial settings of an iTach serial port."""

    baudrate: int
    flow_control: str
    parity: str

    @property
    def suits_seymour(self) -> bool:
        """Return whether a Seymour controller can be reached with these settings."""
        return (
            self.baudrate == SEYMOUR_BAUD_RATE
            and self.flow_control == FLOW_NONE
            and self.parity == PARITY_NO
        )


SEYMOUR_SERIAL_SETTINGS = SerialSettings(SEYMOUR_BAUD_RATE, FLOW_NONE, PARITY_NO)


async def get_serial_settings(
    host: str, port: int = ITACH_COMMAND_PORT, command_timeout: float = DEFAULT_TIMEOUT
) -> SerialSettings:
    """Return the serial settings of an iTach bridge such as the IP2SL."""
    return await _command(host, port, f"get_SERIAL,{SERIAL_CONNECTOR}", command_timeout)


async def set_serial_settings(
    host: str,
    settings: SerialSettings = SEYMOUR_SERIAL_SETTINGS,
    port: int = ITACH_COMMAND_PORT,
    command_timeout: float = DEFAULT_TIMEOUT,
) -> SerialSettings:
    """Change the serial settings of an iTach bridge; return the new settings."""
    command = (
        f"set_SERIAL,{SERIAL_CONNECTOR},{settings.baudrate},{settings.flow_control},"
        f"{settings.parity}"
    )
    return await _command(host, port, command, command_timeout)


async def _command(host: str, port: int, command: str, command_timeout: float) -> SerialSettings:
    """Send a command to the iTach command port and parse the SERIAL reply."""
    try:
        async with asyncio.timeout(command_timeout):
            reader, writer = await asyncio.open_connection(host, port)
            try:
                writer.write(command.encode("ascii") + b"\r")
                await writer.drain()
                reply = await reader.readuntil(b"\r")
            finally:
                writer.close()
    except (OSError, asyncio.IncompleteReadError) as exc:  # TimeoutError is an OSError
        raise SeymourConnectionError(f"iTach command {command!r} failed: {exc!r}") from exc

    text = reply.decode("ascii", errors="replace").strip()
    if not (match := _SERIAL_RE.fullmatch(text)) or match["connector"] != SERIAL_CONNECTOR:
        raise SeymourProtocolError(f"Unexpected reply to {command!r}: {text!r}")
    return SerialSettings(int(match["baudrate"]), match["flow"], match["parity"])
