"""Simple CLI utility to exercise the client."""

import logging
from typing import Annotated

import attrs
import typer
from async_typer import AsyncTyper
from rich import print as rprint

from seymourlib.protocol import DiagnosticOption, MotorID, MovementCode, Ratio

from .client import SeymourClient
from .transport import SeymourTransport

app = AsyncTyper(help="Seymour controller CLI")


@attrs.frozen
class GlobalOptions:
    verbose: bool
    host: str
    port: int
    serial_port: str | None


_GLOBAL_OPTIONS: GlobalOptions | None = None


@app.callback()  # type: ignore
def handle_global_options(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Hostname or IP address for TCP transport.",
            envvar="SEYMOUR_HOST",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port number for TCP transport.",
            envvar="SEYMOUR_PORT",
        ),
    ] = 4999,
    serial_port: Annotated[
        str | None,
        typer.Option(
            "--serial-port",
            "-s",
            help="Serial port device for Serial transport (overrides host/port).",
            envvar="SEYMOUR_SERIAL_PORT",
        ),
    ] = None,
) -> None:
    global _GLOBAL_OPTIONS
    _GLOBAL_OPTIONS = GlobalOptions(verbose=verbose, host=host, port=port, serial_port=serial_port)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)


def _get_client() -> SeymourClient:
    assert _GLOBAL_OPTIONS is not None
    if _GLOBAL_OPTIONS.serial_port:
        from .transport import SerialTransport

        transport: SeymourTransport = SerialTransport(_GLOBAL_OPTIONS.serial_port)
    else:
        from .transport import TCPTransport

        transport = TCPTransport(_GLOBAL_OPTIONS.host, _GLOBAL_OPTIONS.port)

    return SeymourClient(transport)


@app.async_command()  # type: ignore
async def status() -> None:
    """Show the current status code."""
    async with _get_client() as client:
        status = await client.get_status()
        rprint(status)


@app.async_command()  # type: ignore
async def calibrate(motor: Annotated[MotorID, typer.Argument()] = MotorID.ALL) -> None:
    """Calibrate the given motor(s) by moving all the way in & out."""
    async with _get_client() as client:
        await client.calibrate(motor)


positions_app = AsyncTyper(help="Commands for the motors' absolute positions.")
app.add_typer(positions_app, name="positions")


@positions_app.async_command("get")  # type: ignore
async def positions_get() -> None:
    """Show the current motor positions."""
    async with _get_client() as client:
        positions = await client.get_positions()
        for position in positions:
            rprint(position)


@positions_app.async_command("halt")  # type: ignore
async def positions_halt(motor: Annotated[MotorID, typer.Argument()] = MotorID.ALL) -> None:
    """Stop the specified motor(s) at their current position."""
    async with _get_client() as client:
        await client.halt(motor)


@positions_app.async_command("home")  # type: ignore
async def positions_home(motor: Annotated[MotorID, typer.Argument()] = MotorID.ALL) -> None:
    """Move the specified motor(s) to their home position."""
    async with _get_client() as client:
        await client.home(motor)


def _parse_increment(move: bool, jog: bool, until_limit: bool) -> MovementCode:
    match (move, jog, until_limit):
        case (True, False, False):
            return MovementCode.MOVE
        case (False, True, False):
            return MovementCode.JOG
        case (False, False, True):
            return MovementCode.UNTIL_LIMIT
        case _:
            raise typer.BadParameter(
                "Exactly one of --move, --jog, or --until-limit must be specified."
            )


@positions_app.async_command("in")  # type: ignore
async def positions_in(
    motor: Annotated[MotorID, typer.Argument()] = MotorID.ALL,
    move: bool = True,
    jog: bool = False,
    until_limit: bool = False,
) -> None:
    """Move the specified motor(s) inward by the given increment."""
    increment = _parse_increment(move, jog, until_limit)
    async with _get_client() as client:
        await client.move_in(motor, increment)


@positions_app.async_command("out")  # type: ignore
async def positions_out(
    motor: Annotated[MotorID, typer.Argument()] = MotorID.ALL,
    move: bool = True,
    jog: bool = False,
    until_limit: bool = False,
) -> None:
    """Move the specified motor(s) outward by the given increment."""
    increment = _parse_increment(move, jog, until_limit)
    async with _get_client() as client:
        await client.move_out(motor, increment)


preset_app = AsyncTyper(help="Commands for motor presets.")
app.add_typer(preset_app, name="preset")


def _parse_ratio_id(ratio_id: str) -> Ratio:
    try:
        return Ratio(ratio_id)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid ratio ID: {ratio_id}") from exc


@preset_app.async_command("apply")  # type: ignore
async def preset_apply(
    ratio_id: Annotated[str, typer.Argument(help="3-digit ratio ID, e.g. '235' for 2.35:1")],
) -> None:
    """Move motors to the designated ratio preset."""
    ratio = _parse_ratio_id(ratio_id)
    async with _get_client() as client:
        await client.move_to_ratio(ratio)


@preset_app.async_command("list")  # type: ignore
async def preset_list() -> None:
    """List all stored ratio presets."""
    async with _get_client() as client:
        settings = await client.get_ratio_settings()
        for preset in settings:
            rprint(preset)


@preset_app.async_command("reset")  # type: ignore
async def preset_reset(
    ratio_id: Annotated[
        str | None,
        typer.Argument(help="3-digit ratio ID, e.g. '235' for 2.35:1.  Default: ALL PRESETS"),
    ] = None,
) -> None:
    """Restore the given ratio preset(s) to their factory default."""
    ratio = _parse_ratio_id(ratio_id) if ratio_id else None
    ratio_text = ratio if ratio else "ALL PRESETS"
    typer.confirm(f"Are you sure you want to reset {ratio_text} to factory default?", abort=True)
    async with _get_client() as client:
        await client.reset_factory_default(ratio)


@preset_app.async_command("store")  # type: ignore
async def preset_store(
    ratio_id: Annotated[str, typer.Argument(help="3-digit ratio ID, e.g. '235' for 2.35:1")],
) -> None:
    """Store the current motor positions as the designated ratio preset."""
    ratio = _parse_ratio_id(ratio_id)
    async with _get_client() as client:
        await client.update_ratio(ratio)


system_app = AsyncTyper(help="Commands for reading system internals.")
app.add_typer(system_app, name="system")


@system_app.async_command("info")  # type: ignore
async def system_info() -> None:
    """Show static information about the Seymour screen system."""
    async with _get_client() as client:
        info = await client.get_system_info()
        for key, value in attrs.asdict(info).items():
            rprint(f"[bold yellow]{key}[/bold yellow]: {value}")


@system_app.async_command()  # type: ignore
async def diagnostics(option: DiagnosticOption) -> None:
    """Show diagnostic information from the Seymour screen system."""
    async with _get_client() as client:
        diag = await client.get_diagnostics(option)
        rprint(diag)


def main() -> None:
    app()
