"""Tests proving that SeymourClient serialises send/receive pairs so that
concurrent callers never observe torn reads or writes.

Background
----------
RS-232 (and the IP2SL bridge) is strictly half-duplex.  If two callers
interleave their send+receive pairs the byte stream gets mangled:

    Task A  send [01S]          ──▶  device receives [01S]
    Task B  send [01P]          ──▶  device receives [01P]
    Task A  receive()  ◀──  device writes [01P178]   ← WRONG answer for A!
    Task B  receive()  ◀──  device writes [01SP]     ← WRONG answer for B!

Before commit 70b8f2d the lock lived in SeymourTransport.send() only,
leaving the receive path unprotected.  The fix moved the lock into
SeymourClient._execute_operation() so that each send+receive pair runs
atomically.

These tests exercise the fix from several angles:

1. Operation ordering: send/recv pairs are strictly sequential.
2. Real-decoder correctness: concurrent get_status / get_positions
   calls never produce SeymourProtocolError.
3. Mixed fire-and-forget + query operations.
4. High fan-out stress test.
5. Simulated torn-read scenario using a real StreamReader.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from seymourlib import protocol
from seymourlib.client import SeymourClient
from seymourlib.exceptions import SeymourProtocolError
from seymourlib.transport import SeymourTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Realistic canned frames the "device" would return for each command.
STATUS_REQ = protocol.encode_status()  # [01S]
STATUS_RESP = b"[01P178]"  # stopped at ratio 178

POSITIONS_REQ = protocol.encode_positions()  # [01P]
POSITIONS_RESP = b"[012T50.0B25.0]"  # 2 motors

MOVE_OUT_REQ = protocol.encode_move_out(  # fire-and-forget
    protocol.MotorID.TOP, protocol.MovementCode.JOG
)


class OrderTrackingTransport(SeymourTransport):
    """Mock transport that records operation ordering and returns the correct
    response for each request frame.

    A small ``asyncio.sleep`` in both ``send`` and ``receive`` widens the
    window in which interleaving *would* occur without the client-level lock.

    The response for each ``send`` is remembered so that the immediately
    following ``receive`` returns the matching frame — exactly how a real
    half-duplex serial device behaves under the client lock.
    """

    def __init__(
        self,
        response_map: dict[bytes, bytes],
        *,
        op_delay: float = 0.005,
    ) -> None:
        super().__init__()
        self._response_map = response_map
        self._op_delay = op_delay

        # Each entry is ("send", frame) or ("recv", frame).
        self.log: list[tuple[str, bytes]] = []
        # The response for the most recent send; read by the next receive.
        self._last_response: bytes = b""

    async def connect(self) -> None:
        pass

    async def send(self, data: bytes) -> None:
        self.log.append(("send", data))
        # Stash the response that corresponds to the request just sent.
        # Because the client lock guarantees at most one send/receive pair
        # in-flight, a simple instance variable is safe here.
        self._last_response = self._response_map.get(data, b"[01P]")
        await asyncio.sleep(self._op_delay)

    async def receive(self) -> bytes:
        await asyncio.sleep(self._op_delay)
        resp = self._last_response
        self.log.append(("recv", resp))
        return resp


class PipedTransport(SeymourTransport):
    """Transport backed by a real ``asyncio.StreamReader`` so that concurrent
    ``receive()`` calls truly race on the same byte stream — exactly the
    scenario that caused torn reads before the client-level lock was added.

    The "device" side is driven by :meth:`device_write`, which pushes raw
    bytes into the pipe.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device_writer: asyncio.StreamWriter | None = None
        self._frames_sent: list[bytes] = []
        self._response_map: dict[bytes, bytes] = {}

    async def connect(self) -> None:
        # Create an in-process TCP pipe so we get a real StreamReader.
        self.reader = asyncio.StreamReader()
        # We'll feed data directly via feed_data / feed_eof.

    async def send(self, data: bytes) -> None:
        self._frames_sent.append(data)
        # Simulate the device echoing back the mapped response.
        resp = self._response_map.get(data, b"[01P]")
        await asyncio.sleep(0.002)  # simulate wire delay
        assert self.reader is not None
        self.reader.feed_data(resp)

    async def receive(self) -> bytes:
        assert self.reader is not None
        data = await self.reader.readuntil(b"]")
        return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentOperationOrdering:
    """Verify that send/receive pairs are strictly sequential under the
    client-level lock.
    """

    async def test_pairs_never_interleave(self) -> None:
        """With N concurrent request–response operations, the operation log
        must always show alternating (send, recv) pairs — never two
        consecutive sends followed by two receives.
        """
        response_map = {STATUS_REQ: STATUS_RESP, POSITIONS_REQ: POSITIONS_RESP}
        transport = OrderTrackingTransport(response_map)
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        async def do_status() -> None:
            await client.get_status()

        async def do_positions() -> None:
            await client.get_positions()

        tasks = [
            asyncio.create_task(do_status()),
            asyncio.create_task(do_positions()),
            asyncio.create_task(do_status()),
            asyncio.create_task(do_positions()),
        ]
        await asyncio.gather(*tasks)

        # The log must consist of (send, recv) pairs without interleaving.
        assert len(transport.log) == 8  # 4 sends + 4 receives
        for i in range(0, len(transport.log), 2):
            assert transport.log[i][0] == "send", (
                f"Entry {i} should be 'send', got {transport.log[i]}"
            )
            assert transport.log[i + 1][0] == "recv", (
                f"Entry {i + 1} should be 'recv', got {transport.log[i + 1]}"
            )

        await client.close()

    async def test_send_only_ops_serialized_with_queries(self) -> None:
        """Fire-and-forget commands (move_out) must also be serialized with
        query commands so they don't steal bytes from the wire.
        """
        response_map = {STATUS_REQ: STATUS_RESP}
        transport = OrderTrackingTransport(response_map)
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        tasks = [
            asyncio.create_task(client.get_status()),
            asyncio.create_task(client.move_out(protocol.MotorID.TOP, protocol.MovementCode.JOG)),
            asyncio.create_task(client.get_status()),
        ]
        await asyncio.gather(*tasks)

        # Every send must be immediately followed by its recv (if it has one)
        # or the next send (if fire-and-forget).  No two recv in a row.
        sends = [e for e in transport.log if e[0] == "send"]
        recvs = [e for e in transport.log if e[0] == "recv"]

        # 3 sends total: 2 status queries + 1 move_out
        assert len(sends) == 3
        # 2 receives: only the status queries read back
        assert len(recvs) == 2

        # The log should never have two recvs in a row
        for i in range(len(transport.log) - 1):
            if transport.log[i][0] == "recv":
                assert transport.log[i + 1][0] != "recv", (
                    f"Two consecutive recv at index {i}: {transport.log}"
                )

        await client.close()


class TestConcurrentParsing:
    """Concurrent API calls must never produce ``SeymourProtocolError`` due
    to response frames ending up at the wrong decoder.
    """

    async def test_concurrent_get_status(self) -> None:
        """Many concurrent get_status() calls all decode without error."""
        transport = OrderTrackingTransport({STATUS_REQ: STATUS_RESP})
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        results = await asyncio.gather(*[client.get_status() for _ in range(10)])

        for r in results:
            assert isinstance(r, protocol.MaskStatus)
            assert r.code == protocol.StatusCode.STOPPED_AT_RATIO
            assert r.ratio == protocol.Ratio("178")

        await client.close()

    async def test_concurrent_get_positions(self) -> None:
        """Many concurrent get_positions() calls all decode without error."""
        transport = OrderTrackingTransport({POSITIONS_REQ: POSITIONS_RESP})
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        results = await asyncio.gather(*[client.get_positions() for _ in range(10)])

        for r in results:
            assert isinstance(r, list)
            assert len(r) == 2
            assert r[0].motor_id == protocol.MotorID.TOP
            assert r[1].motor_id == protocol.MotorID.BOTTOM

        await client.close()

    async def test_concurrent_mixed_operations_no_parse_error(self) -> None:
        """Interleaving get_status and get_positions — whose response formats
        are incompatible — must never cause a parse error.

        This is the exact scenario that broke before the client-level lock:
        get_positions' response would be fed to decode_status (or vice versa),
        raising ``SeymourProtocolError``.
        """
        response_map = {STATUS_REQ: STATUS_RESP, POSITIONS_REQ: POSITIONS_RESP}
        transport = OrderTrackingTransport(response_map, op_delay=0.01)
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        tasks: list[asyncio.Task[object]] = []
        for i in range(20):
            if i % 2 == 0:
                tasks.append(asyncio.create_task(client.get_status()))
            else:
                tasks.append(asyncio.create_task(client.get_positions()))

        results = await asyncio.gather(*tasks)

        statuses = [r for r in results if isinstance(r, protocol.MaskStatus)]
        positions = [r for r in results if isinstance(r, list)]
        assert len(statuses) == 10
        assert len(positions) == 10

        await client.close()


class TestTornReadPrevention:
    """Use a ``PipedTransport`` backed by a real ``asyncio.StreamReader`` to
    prove that the client lock prevents torn reads.

    Without the lock, two concurrent ``readuntil(b"]")`` calls on the same
    StreamReader race: the first one might consume the ``]`` delimiter of the
    second caller's frame, yielding garbage to both.
    """

    async def test_piped_concurrent_reads_intact(self) -> None:
        """Concurrent queries through a real StreamReader each get a
        well-formed, parseable response.
        """
        transport = PipedTransport()
        transport._response_map = {
            STATUS_REQ: STATUS_RESP,
            POSITIONS_REQ: POSITIONS_RESP,
        }
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        status_task = asyncio.create_task(client.get_status())
        positions_task = asyncio.create_task(client.get_positions())

        status, positions = await asyncio.gather(status_task, positions_task)

        assert isinstance(status, protocol.MaskStatus)
        assert status.code == protocol.StatusCode.STOPPED_AT_RATIO
        assert isinstance(positions, list)
        assert len(positions) == 2

        await client.close()

    async def test_piped_high_fanout(self) -> None:
        """Many concurrent operations on a real StreamReader all succeed."""
        transport = PipedTransport()
        transport._response_map = {STATUS_REQ: STATUS_RESP}
        client = SeymourClient(transport, max_retries=1, request_timeout=5.0)
        await client.connect()

        results = await asyncio.gather(*[client.get_status() for _ in range(20)])

        for r in results:
            assert isinstance(r, protocol.MaskStatus)
            assert r.code == protocol.StatusCode.STOPPED_AT_RATIO

        await client.close()


class TestStressHighFanout:
    """Stress test with many concurrent callers of different operation types."""

    async def test_50_concurrent_mixed_ops(self) -> None:
        """50 concurrent tasks mixing queries and fire-and-forget commands.

        Every query must decode correctly; no ``SeymourProtocolError`` or
        ``AssertionError`` is acceptable.
        """
        response_map = {
            STATUS_REQ: STATUS_RESP,
            POSITIONS_REQ: POSITIONS_RESP,
            MOVE_OUT_REQ: b"IGNORED",  # not read; fire-and-forget
        }
        transport = OrderTrackingTransport(response_map, op_delay=0.002)
        client = SeymourClient(transport, max_retries=1, request_timeout=10.0)
        await client.connect()

        async def status() -> protocol.MaskStatus:
            return await client.get_status()

        async def positions() -> list[protocol.MaskPosition]:
            return await client.get_positions()

        async def move() -> None:
            await client.move_out(protocol.MotorID.TOP, protocol.MovementCode.JOG)

        tasks: list[asyncio.Task[object]] = []
        for i in range(50):
            match i % 3:
                case 0:
                    tasks.append(asyncio.create_task(status()))
                case 1:
                    tasks.append(asyncio.create_task(positions()))
                case 2:
                    tasks.append(asyncio.create_task(move()))

        results = await asyncio.gather(*tasks)

        # Count successful results by type.
        n_status = sum(1 for r in results if isinstance(r, protocol.MaskStatus))
        n_positions = sum(1 for r in results if isinstance(r, list))
        n_none = sum(1 for r in results if r is None)

        assert n_status == 17  # ceil(50/3)
        assert n_positions == 17
        assert n_none == 16
        assert n_status + n_positions + n_none == 50

        await client.close()

    async def test_no_protocol_errors_under_contention(self) -> None:
        """Specifically assert that ``SeymourProtocolError`` is never raised
        when many callers hit the client simultaneously.

        This is a regression test: before the client-level lock, responses
        routinely ended up at the wrong decoder.
        """
        response_map = {STATUS_REQ: STATUS_RESP, POSITIONS_REQ: POSITIONS_RESP}
        transport = OrderTrackingTransport(response_map, op_delay=0.003)
        client = SeymourClient(transport, max_retries=1, request_timeout=10.0)
        await client.connect()

        errors: list[Exception] = []

        async def safe_status() -> protocol.MaskStatus | None:
            try:
                return await client.get_status()
            except SeymourProtocolError as exc:
                errors.append(exc)
                return None

        async def safe_positions() -> list[protocol.MaskPosition] | None:
            try:
                return await client.get_positions()
            except SeymourProtocolError as exc:
                errors.append(exc)
                return None

        tasks: list[asyncio.Task[object]] = []
        for i in range(30):
            if i % 2 == 0:
                tasks.append(asyncio.create_task(safe_status()))
            else:
                tasks.append(asyncio.create_task(safe_positions()))

        results = await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Got {len(errors)} protocol error(s): {errors}"
        assert all(r is not None for r in results)

        await client.close()
