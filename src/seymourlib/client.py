"""Main async client for interacting with the Seymour controller."""

from __future__ import annotations

import asyncio
import logging
import time

from tenacity import (
    AsyncRetrying,
    after_log,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import protocol
from .exceptions import SeymourConnectionError, SeymourProtocolError, SeymourTransportError
from .transport import SeymourTransport

_LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 10.0
HEALTH_CHECK_INTERVAL = 90.0
HEALTH_CHECK_TIMEOUT = 3.0


class SeymourClient:
    def __init__(
        self,
        transport: SeymourTransport,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.transport = transport
        self._lock = asyncio.Lock()
        self._connected = False
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._last_successful_operation = 0.0
        self._health_check_task: asyncio.Task[None] | None = None

    async def _connect_with_retry(self) -> None:
        """Connect with tenacity-managed retry logic."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((SeymourTransportError, ConnectionError, OSError)),
            before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
            after=after_log(_LOGGER, logging.INFO),
        )

        async for attempt in retryer:
            with attempt:
                await self.transport.connect()
                self._connected = True
                self._last_successful_operation = time.time()

    async def connect(self) -> None:
        async with self._lock:
            if self._connected:
                return

            try:
                await self._connect_with_retry()
            except Exception as exc:
                self._connected = False
                raise SeymourConnectionError("Failed to connect after all retries") from exc

    async def close(self) -> None:
        async with self._lock:
            # Stop health checking
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None

            # Close transport
            if self._connected:
                await self.transport.close()
                self._connected = False
                _LOGGER.info("Closed Seymour transport")

    async def start_health_monitoring(self, interval: float = HEALTH_CHECK_INTERVAL) -> None:
        """Start background health monitoring for long-running applications like Home Assistant."""
        if self._health_check_task and not self._health_check_task.done():
            return  # Already running

        self._health_check_task = asyncio.create_task(self._health_check_loop(interval))
        _LOGGER.info("Started health monitoring (interval=%.1fs)", interval)

    async def _health_check_loop(self, interval: float) -> None:
        """Background task that periodically checks connection health."""
        while True:
            try:
                await asyncio.sleep(interval)

                # Skip health check if we've had recent successful operations
                if time.time() - self._last_successful_operation < interval / 2:
                    continue

                # Perform lightweight health check
                if self._connected:
                    try:
                        await self._health_check()
                        _LOGGER.debug("Health check passed")
                    except Exception as exc:
                        _LOGGER.warning("Health check failed, marking disconnected: %s", exc)
                        self._connected = False

            except asyncio.CancelledError:
                _LOGGER.debug("Health monitoring cancelled")
                break
            except Exception as exc:
                _LOGGER.error("Unexpected error in health monitoring: %s", exc)
                await asyncio.sleep(interval)  # Continue monitoring despite errors

    async def _health_check(self) -> None:
        """Perform a lightweight health check by requesting status."""
        # Use a shorter timeout for health checks
        original_timeout = self.request_timeout
        self.request_timeout = min(HEALTH_CHECK_TIMEOUT, original_timeout)
        try:
            await self.get_status()
        finally:
            self.request_timeout = original_timeout

    @property
    def is_connected(self) -> bool:
        """Check if the client believes it's connected."""
        return self._connected

    @property
    def connection_stats(self) -> dict[str, float]:
        """Get connection statistics for monitoring."""
        return {
            "last_successful_operation": self._last_successful_operation,
            "time_since_last_success": (
                time.time() - self._last_successful_operation
                if self._last_successful_operation > 0
                else -1
            ),
        }

    async def __aenter__(self) -> SeymourClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def _execute_operation(self, frame: bytes, receive: bool = True) -> bytes | None:
        """Execute a single transport operation with retry logic."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception_type(
                (SeymourTransportError, TimeoutError, ConnectionError, OSError)
            ),
            before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
        )

        async for attempt in retryer:
            with attempt:
                # Ensure connection before operation
                if not self._connected:
                    await self.connect()

                # Execute with timeout
                async with asyncio.timeout(self.request_timeout):
                    await self.transport.send(frame)
                    if receive:
                        response = await self.transport.receive()
                        self._last_successful_operation = time.time()
                        return response
                    else:
                        self._last_successful_operation = time.time()
                        return None

        # This should never be reached due to retry logic, but satisfy type checker
        return None

    async def _send_and_maybe_receive(self, frame: bytes, receive: bool = True) -> bytes | None:
        """Send frame and optionally receive response with automatic reconnection."""
        try:
            logging.debug("Sending frame: %s", frame)
            return await self._execute_operation(frame, receive)
        except Exception as exc:
            # Mark disconnected for any transport error
            self._connected = False
            if isinstance(exc, SeymourTransportError | TimeoutError | ConnectionError | OSError):
                raise SeymourConnectionError("Transport operation failed after retries") from exc
            else:
                raise SeymourProtocolError("Protocol or unexpected error") from exc

    async def _send(self, frame: bytes) -> None:
        """Low-level: send frame without awaiting reply."""
        await self._send_and_maybe_receive(frame, receive=False)

    async def _send_and_receive(self, frame: bytes) -> bytes:
        """Low-level: send frame and await reply frame."""
        raw = await self._send_and_maybe_receive(frame, receive=True)
        logging.debug("Received frame: %s", raw)
        assert raw is not None
        return raw

    # Public API
    async def get_status(self) -> protocol.MaskStatus:
        raw = await self._send_and_receive(protocol.encode_status())
        return protocol.decode_status(raw)

    async def get_positions(self) -> list[protocol.MaskPosition]:
        raw = await self._send_and_receive(protocol.encode_positions())
        return protocol.decode_positions(raw)

    async def get_system_info(self) -> protocol.SystemInfo:
        raw = await self._send_and_receive(protocol.encode_read_sysinfo())
        return protocol.decode_system_info(raw)

    async def get_ratio_settings(self) -> list[protocol.RatioSetting]:
        raw = await self._send_and_receive(protocol.encode_read_settings())
        return protocol.decode_settings(raw)

    async def get_diagnostics(self, option: protocol.DiagnosticOption) -> str:
        raw = await self._send_and_receive(protocol.encode_diagnostics(option))
        return raw.decode("ascii")

    async def move_out(self, motor: protocol.MotorID, movement: protocol.MovementCode) -> None:
        frame = protocol.encode_move_out(motor, movement)
        await self._send(frame)

    async def move_in(self, motor: protocol.MotorID, movement: protocol.MovementCode) -> None:
        frame = protocol.encode_move_in(motor, movement)
        await self._send(frame)

    async def move_to_ratio(self, ratio: protocol.Ratio) -> None:
        frame = protocol.encode_move_ratio(ratio)
        await self._send(frame)

    async def home(self, motor: protocol.MotorID) -> None:
        frame = protocol.encode_home(motor)
        await self._send(frame)

    async def halt(self, motor: protocol.MotorID) -> None:
        frame = protocol.encode_halt(motor)
        await self._send(frame)

    async def calibrate(self, motor: protocol.MotorID) -> None:
        frame = protocol.encode_calibrate(motor)
        await self._send(frame)

    async def update_ratio(self, ratio: protocol.Ratio) -> None:
        frame = protocol.encode_update_ratio(ratio)
        await self._send(frame)

    async def reset_factory_default(self, ratio: protocol.Ratio | None) -> None:
        frame = protocol.encode_clear_settings(ratio)
        await self._send(frame)
