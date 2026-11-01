"""Main async client for interacting with the Seymour controller."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from tenacity import (
    AsyncRetrying,
    RetryError,
    after_log,
    before_sleep_log,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import protocol
from .exceptions import (
    SeymourClientClosedError,
    SeymourConnectionError,
    SeymourProtocolError,
    SeymourTimeoutError,
    SeymourTransportError,
)
from .transport import SeymourTransport

_LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 10.0
# Requests that time out in a row before the connection is replaced.  A
# controller that is off times out on every request, and reconnecting each
# time only churns the bridge's TCP stack; a dead TCP connection is noticed by
# the transport's keepalive instead.
DEFAULT_RECONNECT_AFTER_TIMEOUTS = 3
RETRYABLE_ERRORS = (SeymourTransportError, TimeoutError, ConnectionError, OSError)
HEALTH_CHECK_INTERVAL = 90.0
HEALTH_CHECK_TIMEOUT = 3.0
# Silence that ends a diagnostics response, which has no reliable frame end.
DIAGNOSTICS_IDLE_TIMEOUT = 0.5


def _is_absolute(movement: protocol.MovementCode) -> bool:
    """Return whether repeating a movement ends in the same place."""
    return movement is protocol.MovementCode.UNTIL_LIMIT


class SeymourClient:
    def __init__(
        self,
        transport: SeymourTransport,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        reconnect_after_timeouts: int = DEFAULT_RECONNECT_AFTER_TIMEOUTS,
        confirm_commands: bool = True,
    ):
        self.transport = transport
        self._lock = asyncio.Lock()
        self._connected = False
        self._closed = False
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.reconnect_after_timeouts = reconnect_after_timeouts
        self.confirm_commands = confirm_commands
        self._timed_out_requests = 0
        # Wall clock time for display, monotonic time for intervals
        self._last_successful_operation = 0.0
        self._last_success_monotonic: float | None = None
        self._health_check_task: asyncio.Task[None] | None = None

    async def _connect_with_retry(self) -> None:
        """Connect with tenacity-managed retry logic."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((SeymourTransportError, ConnectionError, OSError)),
            before_sleep=before_sleep_log(_LOGGER, logging.DEBUG),
            after=after_log(_LOGGER, logging.DEBUG),
        )

        async for attempt in retryer:
            with attempt:
                await self.transport.connect()
                self._connected = True
                self._mark_success()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise SeymourClientClosedError("The client is closed")

    async def connect(self) -> None:
        async with self._lock:
            self._raise_if_closed()
            if self._connected:
                return

            try:
                await self._connect_with_retry()
            except Exception as exc:
                self._connected = False
                raise SeymourConnectionError("Failed to connect after all retries") from exc

    async def disconnect(self) -> None:
        """Drop the connection; the next request opens a new one.

        Use this to recycle a connection that seems stuck.  Unlike close(),
        the client stays usable.
        """
        async with self._lock:
            await self._drop_connection()

    async def close(self) -> None:
        """Close the client for good.

        Requests in flight, such as one waiting to retry, fail at their next
        attempt instead of reconnecting, and later requests fail at once.
        Otherwise a request that outlives its owner would open a connection
        nobody closes, which holds the only connection an IP2SL allows.
        """
        self._closed = True
        async with self._lock:
            # Stop health checking
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None

            await self._drop_connection()

    async def _drop_connection(self) -> None:
        """Close the transport; the caller must hold the lock."""
        # Always close the transport: a failed request clears _connected
        # but leaves the socket open, and an IP2SL only accepts one
        # connection per serial port, so a leaked socket blocks every
        # later connect.  Closing an already-closed transport is a no-op.
        was_connected = self._connected
        self._connected = False
        await self.transport.close()
        if was_connected:
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
                if (
                    self._last_success_monotonic is not None
                    and time.monotonic() - self._last_success_monotonic < interval / 2
                ):
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
        hc_timeout = min(HEALTH_CHECK_TIMEOUT, self.request_timeout)
        raw = await self._send_and_receive(protocol.encode_status(), op_timeout=hc_timeout)
        protocol.decode_status(raw)

    def _mark_success(self) -> None:
        self._timed_out_requests = 0
        self._last_successful_operation = time.time()
        self._last_success_monotonic = time.monotonic()

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
                time.monotonic() - self._last_success_monotonic
                if self._last_success_monotonic is not None
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

    async def _execute_operation(
        self,
        frame: bytes,
        receive: bool = True,
        op_timeout: float | None = None,
        until_idle: float | None = None,
        idempotent: bool = True,
    ) -> bytes | None:
        """Execute a single transport operation with retry logic.

        The lock serialises access so that only one send/receive pair is
        in-flight at a time.  RS-232 (and the IP2SL bridge) is strictly
        half-duplex: interleaving requests causes torn reads where one
        caller consumes bytes belonging to the other's response.

        A frame that isn't idempotent, such as a relative move, is only
        retried while it can't have reached the controller: resending one
        that may have arrived would apply it twice.
        """
        effective_timeout = op_timeout if op_timeout is not None else self.request_timeout
        may_have_sent = False

        def mark_sent() -> None:
            nonlocal may_have_sent
            may_have_sent = True

        def should_retry(exc: BaseException) -> bool:
            return isinstance(exc, RETRYABLE_ERRORS) and (idempotent or not may_have_sent)

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception(should_retry),
            # Retries are routine; callers report requests that finally fail
            before_sleep=before_sleep_log(_LOGGER, logging.DEBUG),
        )

        async for attempt in retryer:
            with attempt:
                # The lock is only held for one attempt, never while waiting
                # to retry, so other requests, such as a halt, get their turn.
                async with self._lock:
                    self._raise_if_closed()
                    try:
                        return await self._attempt(
                            frame, receive, effective_timeout, until_idle, mark_sent
                        )
                    except TimeoutError:
                        raise
                    except (SeymourTransportError, OSError):
                        # The connection is broken, so retrying on it can't
                        # succeed; the next attempt opens a new one.
                        await self._drop_connection()
                        raise
                    except SeymourProtocolError:
                        raise
                    except Exception:
                        # Nothing is known about the state of the connection
                        # after an unexpected error, so don't trust it again.
                        await self._drop_connection()
                        raise

        # This should never be reached due to retry logic, but satisfy type checker
        return None

    async def _attempt(
        self,
        frame: bytes,
        receive: bool,
        op_timeout: float,
        until_idle: float | None,
        mark_sent: Callable[[], None],
    ) -> bytes | None:
        """Send a frame once, connecting first if needed; the caller holds the lock."""
        if not self._connected:
            try:
                await self.transport.connect()
            except (TimeoutError, OSError) as exc:
                # Keep a connect timeout apart from a controller that doesn't answer
                raise SeymourTransportError(f"Failed to connect: {exc!r}") from exc
            self._connected = True
            self._mark_success()

        # Drain any stale bytes from the buffer before sending.
        # This prevents a desynchronised buffer (e.g. from a
        # previous timed-out request) from causing the next
        # receive() to return the wrong frame.
        #
        # This is safe because the Seymour protocol does not
        # have any unsolicited frames or notifications.
        drained = await self.transport.drain_read_buffer()
        if drained:
            # Whitespace is a line terminator after the previous
            # frame, not a sign of a desynchronised stream.
            _LOGGER.log(
                logging.WARNING if drained.strip() else logging.DEBUG,
                "Drained %d stale bytes from buffer before send: %r",
                len(drained),
                drained[:80],
            )

        async with asyncio.timeout(op_timeout):
            mark_sent()
            await self.transport.send(frame)
            if not receive:
                if self.confirm_commands:
                    # Commands get no reply, and a write to a dead connection
                    # still succeeds locally; any answer to a status request
                    # on the same connection proves the command was delivered.
                    await self.transport.send(protocol.encode_status())
                    await self.transport.receive()
                self._mark_success()
                return None
            response = await (
                self.transport.receive()
                if until_idle is None
                else self.transport.receive_until_idle(until_idle)
            )
            self._mark_success()
            return response

    async def _send_and_maybe_receive(
        self,
        frame: bytes,
        receive: bool = True,
        op_timeout: float | None = None,
        until_idle: float | None = None,
        idempotent: bool = True,
    ) -> bytes | None:
        """Send frame and optionally receive response with automatic reconnection."""
        try:
            _LOGGER.debug("Sending frame: %s", frame)
            return await self._execute_operation(
                frame,
                receive,
                op_timeout=op_timeout,
                until_idle=until_idle,
                idempotent=idempotent,
            )
        except (
            SeymourTransportError,
            RetryError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            # Only mark disconnected for transport-level failures.
            # Protocol errors (malformed response) do NOT indicate a
            # broken connection and should not trigger reconnection.
            cause = exc.last_attempt.exception() if isinstance(exc, RetryError) else exc
            if isinstance(cause, TimeoutError):
                # The connection itself may be fine, so only replace it when
                # the silence persists.
                self._timed_out_requests += 1
                if self._timed_out_requests >= self.reconnect_after_timeouts:
                    self._timed_out_requests = 0
                    self._connected = False
                raise SeymourTimeoutError("No response from the controller") from exc
            self._connected = False
            raise SeymourConnectionError("Transport operation failed after retries") from exc
        except SeymourClientClosedError:
            raise
        except SeymourProtocolError:
            # The connection is fine; the device sent a parseable but
            # unexpected frame.  Re-raise without touching _connected.
            raise
        except Exception as exc:
            raise SeymourConnectionError("Unexpected error; the connection was reset") from exc

    async def _send(self, frame: bytes, idempotent: bool = True) -> None:
        """Low-level: send frame without awaiting reply."""
        await self._send_and_maybe_receive(frame, receive=False, idempotent=idempotent)

    async def _send_and_receive(
        self,
        frame: bytes,
        op_timeout: float | None = None,
        until_idle: float | None = None,
    ) -> bytes:
        """Low-level: send frame and await reply frame."""
        raw = await self._send_and_maybe_receive(
            frame, receive=True, op_timeout=op_timeout, until_idle=until_idle
        )
        _LOGGER.debug("Received frame: %s", raw)
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
        raw = await self._send_and_receive(
            protocol.encode_diagnostics(option), until_idle=DIAGNOSTICS_IDLE_TIMEOUT
        )
        return raw.decode("ascii", errors="replace")

    async def move_out(self, motor: protocol.MotorID, movement: protocol.MovementCode) -> None:
        frame = protocol.encode_move_out(motor, movement)
        await self._send(frame, idempotent=_is_absolute(movement))

    async def move_in(self, motor: protocol.MotorID, movement: protocol.MovementCode) -> None:
        frame = protocol.encode_move_in(motor, movement)
        await self._send(frame, idempotent=_is_absolute(movement))

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
