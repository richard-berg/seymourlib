"""Comprehensive test suite for seymourlib.client module."""

# mypy: disable-error-code="unreachable"

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from seymourlib import protocol
from seymourlib.client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    SeymourClient,
)
from seymourlib.exceptions import (
    SeymourConnectionError,
    SeymourProtocolError,
    SeymourTransportError,
)
from seymourlib.transport import SeymourTransport


class MockTransport(SeymourTransport):
    """Mock transport for testing SeymourClient."""

    def __init__(self) -> None:
        super().__init__()
        self.connect_mock = AsyncMock()
        self.close_mock = AsyncMock()
        self.send_mock = AsyncMock()
        self.receive_mock = AsyncMock()
        self.receive_responses: list[bytes] = []
        self._receive_index = 0

    async def connect(self) -> None:
        await self.connect_mock()

    async def close(self) -> None:
        await self.close_mock()

    async def send(self, data: bytes) -> None:
        await self.send_mock(data)

    async def receive(self) -> bytes:
        if self.receive_responses and self._receive_index < len(self.receive_responses):
            response = self.receive_responses[self._receive_index]
            self._receive_index += 1
            return response
        result = await self.receive_mock()
        return result if isinstance(result, bytes) else b"default_response"

    def reset_mocks(self) -> None:
        """Reset all mocks and response index."""
        self.connect_mock.reset_mock()
        self.close_mock.reset_mock()
        self.send_mock.reset_mock()
        self.receive_mock.reset_mock()
        self._receive_index = 0

    def set_receive_responses(self, responses: list[bytes]) -> None:
        """Set responses to return from receive calls."""
        self.receive_responses = responses
        self._receive_index = 0


class TestSeymourClient:
    """Test SeymourClient initialization and basic properties."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport)

    def test_initialization_defaults(self, transport: MockTransport) -> None:
        """Test client initialization with default parameters."""
        client = SeymourClient(transport)
        assert client.transport is transport
        assert client.max_retries == DEFAULT_MAX_RETRIES
        assert client.request_timeout == DEFAULT_REQUEST_TIMEOUT
        assert not client.is_connected
        assert client._last_successful_operation == 0.0
        assert client._health_check_task is None

    def test_initialization_custom(self, transport: MockTransport) -> None:
        """Test client initialization with custom parameters."""
        client = SeymourClient(transport, max_retries=5, request_timeout=20.0)
        assert client.transport is transport
        assert client.max_retries == 5
        assert client.request_timeout == 20.0
        assert not client.is_connected

    def test_connection_stats_initial(self, client: SeymourClient) -> None:
        """Test connection stats before any operations."""
        stats = client.connection_stats
        assert stats["last_successful_operation"] == 0.0
        assert stats["time_since_last_success"] == -1

    @patch("time.time")
    def test_connection_stats_after_operation(self, mock_time: Mock, client: SeymourClient) -> None:
        """Test connection stats after successful operation."""
        mock_time.return_value = 1000.0
        client._last_successful_operation = 950.0

        stats = client.connection_stats
        assert stats["last_successful_operation"] == 950.0
        assert stats["time_since_last_success"] == 50.0


class TestSeymourClientConnection:
    """Test connection management functionality."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport)

    @pytest.mark.asyncio
    async def test_connect_success(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test successful connection."""
        assert not client.is_connected

        await client.connect()

        assert client.is_connected
        transport.connect_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test connection when already connected."""
        await client.connect()  # First connection
        transport.connect_mock.reset_mock()

        await client.connect()  # Second connection attempt

        assert client.is_connected
        transport.connect_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_transport_error(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test connection failure due to transport error."""
        transport.connect_mock.side_effect = SeymourTransportError("Connection failed")

        with pytest.raises(SeymourConnectionError, match="Failed to connect after all retries"):
            await client.connect()

        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_connect_with_retries(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test connection retries on failure."""
        # Fail twice, then succeed
        transport.connect_mock.side_effect = [
            ConnectionError("First failure"),
            OSError("Second failure"),
            None,  # Success
        ]

        await client.connect()

        assert client.is_connected
        assert transport.connect_mock.call_count == 3

    @pytest.mark.asyncio
    async def test_close_success(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test successful close."""
        await client.connect()
        assert client.is_connected

        await client.close()

        assert not client.is_connected
        transport.close_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_not_connected(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test close when not connected."""
        assert not client.is_connected

        await client.close()

        assert not client.is_connected
        transport.close_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_health_monitoring(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test close cancels health monitoring task."""
        await client.connect()
        await client.start_health_monitoring(interval=0.1)

        # Ensure health check task is running
        assert client._health_check_task is not None
        assert not client._health_check_task.done()

        # Store task reference before closing
        health_task = client._health_check_task

        await client.close()

        # Health check task should be cancelled
        assert health_task.cancelled()
        assert not client.is_connected


class TestSeymourClientContextManager:
    """Test async context manager functionality."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport)

    @pytest.mark.asyncio
    async def test_context_manager_success(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test successful context manager usage."""
        async with client as ctx_client:
            assert ctx_client is client
            assert client.is_connected
            transport.connect_mock.assert_called_once()

        assert not client.is_connected
        transport.close_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exception(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test context manager cleanup after exception."""
        with pytest.raises(ValueError):
            async with client:
                assert client.is_connected
                raise ValueError("Test exception")

        assert not client.is_connected
        transport.close_mock.assert_called_once()


class TestSeymourClientHealthMonitoring:
    """Test health monitoring functionality."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport)

    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, client: SeymourClient) -> None:
        """Test starting health monitoring."""
        await client.connect()

        await client.start_health_monitoring(interval=0.1)

        assert client._health_check_task is not None
        assert not client._health_check_task.done()

        await client.close()

    @pytest.mark.asyncio
    async def test_start_health_monitoring_already_running(self, client: SeymourClient) -> None:
        """Test starting health monitoring when already running."""
        await client.connect()
        await client.start_health_monitoring(interval=0.1)

        task1 = client._health_check_task
        await client.start_health_monitoring(interval=0.2)  # Different interval
        task2 = client._health_check_task

        # Should be the same task
        assert task1 is task2

        await client.close()

    @pytest.mark.asyncio
    @patch("seymourlib.client.time.time")
    async def test_health_check_skip_recent_operation(
        self, mock_time: Mock, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test health check skipped when recent successful operation."""
        mock_time.return_value = 1000.0
        client._last_successful_operation = 999.9  # Very recent

        # Mock get_status to track if it's called
        with patch.object(client, "get_status", new_callable=AsyncMock) as mock_get_status:
            await client.connect()
            await client.start_health_monitoring(interval=0.05)  # 50ms interval

            # Wait a bit longer than the interval
            await asyncio.sleep(0.1)

            # get_status should not be called because operation was recent
            mock_get_status.assert_not_called()

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_failure_marks_disconnected(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test health check failure marks client as disconnected."""
        # Mock get_status to fail
        with patch.object(client, "get_status", new_callable=AsyncMock) as mock_get_status:
            mock_get_status.side_effect = SeymourTransportError("Health check failed")

            await client.connect()
            await client.start_health_monitoring(interval=0.05)

            # Wait for health check to run
            await asyncio.sleep(0.1)

            # Client should be marked as disconnected
            assert not client.is_connected

        await client.close()


class TestSeymourClientOperations:
    """Test transport operations and protocol methods."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport, max_retries=2, request_timeout=1.0)

    @pytest.mark.asyncio
    async def test_execute_operation_success(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test successful operation execution."""
        transport.receive_mock.return_value = b"[test_response]"

        await client.connect()
        result = await client._execute_operation(b"[test_frame]", receive=True)

        assert result == b"[test_response]"
        transport.send_mock.assert_called_once_with(b"[test_frame]")
        transport.receive_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_operation_send_only(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test operation execution without receiving response."""
        await client.connect()
        result = await client._execute_operation(b"[test_frame]", receive=False)

        assert result is None
        transport.send_mock.assert_called_once_with(b"[test_frame]")
        transport.receive_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_operation_auto_reconnect(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test operation automatically reconnects when disconnected."""
        client._connected = False  # Start disconnected
        transport.receive_mock.return_value = b"[response]"

        result = await client._execute_operation(b"[frame]", receive=True)

        assert result == b"[response]"
        assert client.is_connected
        # Should have called connect due to disconnected state
        transport.connect_mock.assert_called()

    @pytest.mark.asyncio
    async def test_execute_operation_with_retries(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test operation retries on transport errors."""
        # Fail once, then succeed
        transport.send_mock.side_effect = [SeymourTransportError("Transport error"), None]
        transport.receive_mock.return_value = b"[response]"

        await client.connect()
        result = await client._execute_operation(b"[frame]", receive=True)

        assert result == b"[response]"
        assert transport.send_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_operation_timeout(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test operation timeout handling."""

        async def slow_send(data: bytes) -> None:
            await asyncio.sleep(2.0)  # Longer than request_timeout

        transport.send_mock.side_effect = slow_send
        await client.connect()

        # Should raise RetryError due to tenacity wrapping the TimeoutError
        from tenacity import RetryError

        with pytest.raises(RetryError):
            await client._execute_operation(b"[frame]", receive=False)

    @pytest.mark.asyncio
    async def test_send_and_maybe_receive_marks_disconnected_on_error(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test that transport errors mark client as disconnected."""
        transport.send_mock.side_effect = SeymourTransportError("Transport failed")
        await client.connect()

        with pytest.raises(SeymourConnectionError):
            await client._send_and_maybe_receive(b"[frame]")

        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_send_and_maybe_receive_protocol_error(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test non-transport errors raise protocol error."""
        transport.send_mock.side_effect = ValueError("Protocol error")
        await client.connect()

        with pytest.raises(SeymourProtocolError):
            await client._send_and_maybe_receive(b"[frame]")

        assert not client.is_connected


class TestSeymourClientPublicAPI:
    """Test public API methods."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport)

    @pytest.mark.asyncio
    async def test_get_status(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test get_status method."""
        # Mock protocol encoding/decoding
        with (
            patch("seymourlib.protocol.encode_status") as mock_encode,
            patch("seymourlib.protocol.decode_status") as mock_decode,
        ):
            mock_encode.return_value = b"[S]"
            mock_decode.return_value = protocol.MaskStatus(
                code=protocol.StatusCode.STOPPED_AT_RATIO, ratio=protocol.Ratio("123")
            )
            transport.receive_mock.return_value = b"[S123]"

            await client.connect()
            result = await client.get_status()

            assert isinstance(result, protocol.MaskStatus)
            mock_encode.assert_called_once()
            mock_decode.assert_called_once_with(b"[S123]")

    @pytest.mark.asyncio
    async def test_get_positions(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test get_positions method."""
        with (
            patch("seymourlib.protocol.encode_positions") as mock_encode,
            patch("seymourlib.protocol.decode_positions") as mock_decode,
        ):
            mock_encode.return_value = b"[P]"
            mock_decode.return_value = [
                protocol.MaskPosition(motor_id=protocol.MotorID.TOP, position_pct=50.0)
            ]
            transport.receive_mock.return_value = b"[PT50.0]"

            await client.connect()
            result = await client.get_positions()

            assert isinstance(result, list)
            assert len(result) == 1
            mock_encode.assert_called_once()
            mock_decode.assert_called_once_with(b"[PT50.0]")

    @pytest.mark.asyncio
    async def test_get_system_info(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test get_system_info method."""
        with (
            patch("seymourlib.protocol.encode_read_sysinfo") as mock_encode,
            patch("seymourlib.protocol.decode_system_info") as mock_decode,
        ):
            mock_encode.return_value = b"[Y]"
            mock_info = protocol.SystemInfo(
                screen_model="TEST",
                width_inches=100.0,
                height_inches=50.0,
                serial_number=protocol.Serial("AB", 12, 23, "12345"),
                mask_ids=[protocol.MotorID.TOP, protocol.MotorID.BOTTOM],
            )
            mock_decode.return_value = mock_info
            transport.receive_mock.return_value = b"[Y...]"

            await client.connect()
            result = await client.get_system_info()

            assert result is mock_info
            mock_encode.assert_called_once()
            mock_decode.assert_called_once_with(b"[Y...]")

    @pytest.mark.asyncio
    async def test_get_ratio_settings(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test get_ratio_settings method."""
        with (
            patch("seymourlib.protocol.encode_read_settings") as mock_encode,
            patch("seymourlib.protocol.decode_settings") as mock_decode,
        ):
            mock_encode.return_value = b"[R]"
            mock_decode.return_value = [
                protocol.RatioSetting(
                    ratio=protocol.Ratio("123"),
                    label="Test",
                    width_inches=100.0,
                    height_inches=50.0,
                    motor_positions_pct=[25.0, 75.0],
                    motor_adjustments_pct=[0.0, 0.0],
                )
            ]
            transport.receive_mock.return_value = b"[R...]"

            await client.connect()
            result = await client.get_ratio_settings()

            assert isinstance(result, list)
            assert len(result) == 1
            mock_encode.assert_called_once()
            mock_decode.assert_called_once_with(b"[R...]")

    @pytest.mark.asyncio
    async def test_get_diagnostics(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test get_diagnostics method."""
        with patch("seymourlib.protocol.encode_diagnostics") as mock_encode:
            mock_encode.return_value = b"[@A]"
            transport.receive_mock.return_value = b"Diagnostic info"

            await client.connect()
            result = await client.get_diagnostics(protocol.DiagnosticOption.LIST_FS)

            assert result == "Diagnostic info"
            mock_encode.assert_called_once_with(protocol.DiagnosticOption.LIST_FS)

    @pytest.mark.asyncio
    async def test_move_out(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test move_out method."""
        with patch("seymourlib.protocol.encode_move_out") as mock_encode:
            mock_encode.return_value = b"[OT]"

            await client.connect()
            await client.move_out(protocol.MotorID.TOP, protocol.MovementCode.JOG)

            mock_encode.assert_called_once_with(protocol.MotorID.TOP, protocol.MovementCode.JOG)
            transport.send_mock.assert_called_once_with(b"[OT]")

    @pytest.mark.asyncio
    async def test_move_in(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test move_in method."""
        with patch("seymourlib.protocol.encode_move_in") as mock_encode:
            mock_encode.return_value = b"[IB]"

            await client.connect()
            await client.move_in(protocol.MotorID.BOTTOM, protocol.MovementCode.MOVE)

            mock_encode.assert_called_once_with(protocol.MotorID.BOTTOM, protocol.MovementCode.MOVE)
            transport.send_mock.assert_called_once_with(b"[IB]")

    @pytest.mark.asyncio
    async def test_move_to_ratio(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test move_to_ratio method."""
        with patch("seymourlib.protocol.encode_move_ratio") as mock_encode:
            mock_encode.return_value = b"[M123]"
            ratio = protocol.Ratio("123")

            await client.connect()
            await client.move_to_ratio(ratio)

            mock_encode.assert_called_once_with(ratio)
            transport.send_mock.assert_called_once_with(b"[M123]")

    @pytest.mark.asyncio
    async def test_home(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test home method."""
        with patch("seymourlib.protocol.encode_home") as mock_encode:
            mock_encode.return_value = b"[AL]"

            await client.connect()
            await client.home(protocol.MotorID.LEFT)

            mock_encode.assert_called_once_with(protocol.MotorID.LEFT)
            transport.send_mock.assert_called_once_with(b"[AL]")

    @pytest.mark.asyncio
    async def test_halt(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test halt method."""
        with patch("seymourlib.protocol.encode_halt") as mock_encode:
            mock_encode.return_value = b"[HR]"

            await client.connect()
            await client.halt(protocol.MotorID.RIGHT)

            mock_encode.assert_called_once_with(protocol.MotorID.RIGHT)
            transport.send_mock.assert_called_once_with(b"[HR]")

    @pytest.mark.asyncio
    async def test_calibrate(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test calibrate method."""
        with patch("seymourlib.protocol.encode_calibrate") as mock_encode:
            mock_encode.return_value = b"[CV]"

            await client.connect()
            await client.calibrate(protocol.MotorID.VERTICAL)

            mock_encode.assert_called_once_with(protocol.MotorID.VERTICAL)
            transport.send_mock.assert_called_once_with(b"[CV]")

    @pytest.mark.asyncio
    async def test_update_ratio(self, client: SeymourClient, transport: MockTransport) -> None:
        """Test update_ratio method."""
        with patch("seymourlib.protocol.encode_update_ratio") as mock_encode:
            mock_encode.return_value = b"[U456]"
            ratio = protocol.Ratio("456")

            await client.connect()
            await client.update_ratio(ratio)

            mock_encode.assert_called_once_with(ratio)
            transport.send_mock.assert_called_once_with(b"[U456]")

    @pytest.mark.asyncio
    async def test_reset_factory_default_with_ratio(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test reset_factory_default method with ratio."""
        with patch("seymourlib.protocol.encode_clear_settings") as mock_encode:
            mock_encode.return_value = b"[X789]"
            ratio = protocol.Ratio("789")

            await client.connect()
            await client.reset_factory_default(ratio)

            mock_encode.assert_called_once_with(ratio)
            transport.send_mock.assert_called_once_with(b"[X789]")

    @pytest.mark.asyncio
    async def test_reset_factory_default_no_ratio(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test reset_factory_default method without ratio."""
        with patch("seymourlib.protocol.encode_clear_settings") as mock_encode:
            mock_encode.return_value = b"[X]"

            await client.connect()
            await client.reset_factory_default(None)

            mock_encode.assert_called_once_with(None)
            transport.send_mock.assert_called_once_with(b"[X]")


class TestSeymourClientHealthCheckIntegration:
    """Integration tests for health check functionality."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport, request_timeout=1.0)

    @pytest.mark.asyncio
    async def test_health_check_timeout_modification(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test health check uses a shorter timeout without mutating the instance."""
        original_timeout = client.request_timeout
        assert original_timeout == 1.0

        # Return a valid status frame so decode_status succeeds
        transport.receive_mock.return_value = b"[01P]"  # valid stopped status

        await client.connect()
        await client._health_check()

        # request_timeout must NOT have been mutated
        assert client.request_timeout == original_timeout

    @pytest.mark.asyncio
    async def test_health_check_timeout_restoration_on_exception(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test timeout is not mutated even if health check fails."""
        original_timeout = client.request_timeout

        transport.send_mock.side_effect = SeymourTransportError("Health check failed")

        await client.connect()

        with pytest.raises(SeymourConnectionError):
            await client._health_check()

        # Timeout must still be the original value
        assert client.request_timeout == original_timeout


class TestSeymourClientEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    @pytest.fixture
    def client(self, transport: MockTransport) -> SeymourClient:
        """Create a SeymourClient with mock transport."""
        return SeymourClient(transport, max_retries=1)

    @pytest.mark.asyncio
    async def test_operation_failure_after_max_retries(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test operation failure after exhausting retries."""
        transport.send_mock.side_effect = SeymourTransportError("Persistent failure")
        await client.connect()

        # Should raise RetryError when all retries are exhausted
        from tenacity import RetryError

        with pytest.raises(RetryError):
            await client._execute_operation(b"[frame]")

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test concurrent operations don't interfere with each other."""
        transport.receive_mock.side_effect = [b"[response1]", b"[response2]"]
        await client.connect()

        # Start two operations concurrently
        task1 = asyncio.create_task(client._execute_operation(b"[frame1]"))
        task2 = asyncio.create_task(client._execute_operation(b"[frame2]"))

        results = await asyncio.gather(task1, task2)

        assert len(results) == 2
        assert b"[response1]" in results
        assert b"[response2]" in results

    @pytest.mark.asyncio
    async def test_health_monitoring_with_exceptions(
        self, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test health monitoring continues after transport exceptions."""
        call_count = 0
        original_send = transport.send_mock

        async def failing_send(data: bytes) -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise SeymourTransportError(f"Transient failure {call_count}")
            await original_send(data)

        # Return a valid status frame for successful health checks
        transport.receive_mock.return_value = b"[01P]"

        await client.connect()
        client._last_successful_operation = 0.0  # Force health checks

        transport.send_mock.side_effect = failing_send

        await client.start_health_monitoring(interval=0.05)

        # Wait for multiple health check cycles
        await asyncio.sleep(0.5)

        # Health monitoring should still be running despite exceptions
        assert client._health_check_task is not None and not client._health_check_task.done()
        assert call_count >= 1

        await client.close()

    @pytest.mark.asyncio
    @patch("time.time")
    async def test_last_successful_operation_tracking(
        self, mock_time: Mock, client: SeymourClient, transport: MockTransport
    ) -> None:
        """Test that successful operations update last_successful_operation timestamp."""
        mock_time.return_value = 12345.0
        transport.receive_mock.return_value = b"[response]"

        await client.connect()
        await client._execute_operation(b"[frame]")

        assert client._last_successful_operation == 12345.0
