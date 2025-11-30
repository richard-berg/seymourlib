"""Comprehensive test suite for seymourlib.transport module."""

# mypy: disable-error-code="unreachable"

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from seymourlib.exceptions import SeymourTransportError
from seymourlib.transport import (
    ITACH_SERIAL_PORT,
    SEYMOUR_BAUD_RATE,
    SerialTransport,
    SeymourTransport,
    TCPTransport,
)


class MockTransport(SeymourTransport):
    """Concrete implementation of SeymourTransport for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.reader: AsyncMock | None = None
        self.writer: AsyncMock | None = None
        self.connect_called = False

    async def connect(self) -> None:
        """Mock implementation that sets up mock reader/writer."""
        self.connect_called = True
        self.reader = AsyncMock(spec=asyncio.StreamReader)
        self.writer = AsyncMock(spec=asyncio.StreamWriter)


class TestSeymourTransportBase:
    """Test the abstract SeymourTransport base class functionality."""

    @pytest.fixture
    def transport(self) -> MockTransport:
        """Create a mock transport for testing."""
        return MockTransport()

    def test_initialization(self, transport: MockTransport) -> None:
        """Test transport initialization."""
        assert transport.reader is None
        assert transport.writer is None
        assert transport._lock is not None
        assert isinstance(transport._lock, asyncio.Lock)
        assert not transport.connect_called

    @pytest.mark.asyncio
    async def test_connect_abstract_method(self) -> None:
        """Test that connect is abstract and must be implemented."""
        # Verify the abstract method exists
        assert hasattr(SeymourTransport, "connect")
        assert getattr(SeymourTransport.connect, "__isabstractmethod__", False)

    @pytest.mark.asyncio
    async def test_close_with_active_connection(self, transport: MockTransport) -> None:
        """Test closing an active connection."""
        await transport.connect()
        assert transport.reader is not None
        assert transport.writer is not None

        await transport.close()
        assert transport.reader is None
        assert transport.writer is None

    @pytest.mark.asyncio
    async def test_close_without_connection(self, transport: MockTransport) -> None:
        """Test closing when not connected."""
        # Should not raise an exception
        await transport.close()

        assert transport.reader is None
        assert transport.writer is None

    @pytest.mark.asyncio
    async def test_send_valid_frame(self, transport: MockTransport) -> None:
        """Test sending a valid data frame."""
        await transport.connect()

        test_data = b"[test_frame]"
        await transport.send(test_data)

        # Verify the mock was called correctly
        assert transport.writer is not None
        transport.writer.write.assert_called_once_with(test_data)
        transport.writer.drain.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_without_connection(self, transport: MockTransport) -> None:
        """Test sending data when not connected."""
        test_data = b"[test_frame]"

        with pytest.raises(SeymourTransportError, match="Not connected"):
            await transport.send(test_data)

    @pytest.mark.asyncio
    async def test_send_invalid_frame_start(self, transport: MockTransport) -> None:
        """Test sending data that doesn't start with '['."""
        await transport.connect()

        with pytest.raises(AssertionError, match="Frame must start with '\\['"):
            await transport.send(b"invalid_frame]")

    @pytest.mark.asyncio
    async def test_send_invalid_frame_end(self, transport: MockTransport) -> None:
        """Test sending data that doesn't end with ']'."""
        await transport.connect()

        with pytest.raises(AssertionError, match="Frame must end with '\\]'"):
            await transport.send(b"[invalid_frame")

    @pytest.mark.asyncio
    async def test_send_concurrency_protection(self, transport: MockTransport) -> None:
        """Test that send operations are properly serialized."""
        await transport.connect()
        assert transport.writer is not None
        s = b""

        def write_side_effect(data: bytes) -> None:
            nonlocal s
            s += data

        transport.writer.write.side_effect = write_side_effect

        # Create multiple concurrent send operations
        tasks = [
            transport.send(b"[frame1]"),
            transport.send(b"[frame2]"),
            transport.send(b"[frame3]"),
        ]

        await asyncio.gather(*tasks)

        # All sends should complete successfully
        assert transport.writer.write.call_count == 3
        assert transport.writer.drain.call_count == 3

        # Although frames may arrive in any order, all should be present & not jumbled with others
        assert b"[frame1]" in s
        assert b"[frame2]" in s
        assert b"[frame3]" in s

    @pytest.mark.asyncio
    async def test_receive_valid_frame(self, transport: MockTransport) -> None:
        """Test receiving a complete data frame."""
        await transport.connect()

        expected_data = b"[response_frame]"
        assert transport.reader is not None
        transport.reader.readuntil.return_value = expected_data

        result = await transport.receive()

        assert result == expected_data
        transport.reader.readuntil.assert_called_once_with(b"]")

    @pytest.mark.asyncio
    async def test_receive_without_connection(self, transport: MockTransport) -> None:
        """Test receiving data when not connected."""
        with pytest.raises(SeymourTransportError, match="Not connected"):
            await transport.receive()

    @pytest.mark.asyncio
    async def test_receive_connection_closed(self, transport: MockTransport) -> None:
        """Test handling connection closure during receive."""
        await transport.connect()

        # Configure mock reader to simulate connection closed
        assert transport.reader is not None
        transport.reader.readuntil.side_effect = asyncio.IncompleteReadError(b"partial", 10)

        with pytest.raises(SeymourTransportError, match="Connection closed"):
            await transport.receive()


class TestTCPTransport:
    """Test the TCPTransport implementation."""

    def test_initialization(self) -> None:
        """Test TCP transport initialization."""
        transport = TCPTransport("localhost", 8080)

        assert transport.host == "localhost"
        assert transport.port == 8080
        assert transport.reader is None
        assert transport.writer is None

    def test_initialization_default_port(self) -> None:
        """Test TCP transport initialization with default port."""
        transport = TCPTransport("192.168.1.100")

        assert transport.host == "192.168.1.100"
        assert transport.port == ITACH_SERIAL_PORT
        assert transport.reader is None
        assert transport.writer is None

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful TCP connection."""
        transport = TCPTransport("localhost", 8080)

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_writer = AsyncMock(spec=asyncio.StreamWriter)

        with patch("asyncio.open_connection") as mock_open:
            mock_open.return_value = (mock_reader, mock_writer)

            await transport.connect()

            mock_open.assert_called_once_with("localhost", 8080)
            assert transport.reader is mock_reader
            assert transport.writer is mock_writer

            await transport.close()

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        """Test TCP connection failure."""
        transport = TCPTransport("invalid_host", 8080)

        with patch("asyncio.open_connection") as mock_open:
            mock_open.side_effect = OSError("Connection failed")

            with pytest.raises(OSError, match="Connection failed"):
                await transport.connect()

        assert transport.reader is None
        assert transport.writer is None
        await transport.close()

    @pytest.mark.asyncio
    async def test_full_tcp_workflow(self) -> None:
        """Test complete TCP transport workflow."""
        transport = TCPTransport("localhost", 8080)

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_writer = AsyncMock(spec=asyncio.StreamWriter)

        # Configure mock behaviors
        mock_reader.readuntil.return_value = b"[response]"

        with patch("asyncio.open_connection") as mock_open:
            mock_open.return_value = (mock_reader, mock_writer)

            # Connect
            await transport.connect()

            # Send data
            await transport.send(b"[request]")
            mock_writer.write.assert_called_once_with(b"[request]")
            mock_writer.drain.assert_called_once()

            # Receive data
            response = await transport.receive()
            assert response == b"[response]"
            mock_reader.readuntil.assert_called_once_with(b"]")

            # Close connection
            await transport.close()
            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_called_once()


class TestSerialTransport:
    """Test the SerialTransport implementation."""

    def test_initialization(self) -> None:
        """Test serial transport initialization."""
        transport = SerialTransport("/dev/ttyUSB0", 9600)

        assert transport.port == "/dev/ttyUSB0"
        assert transport.baudrate == 9600
        assert transport.reader is None
        assert transport.writer is None

    def test_initialization_default_baudrate(self) -> None:
        """Test serial transport initialization with default baudrate."""
        transport = SerialTransport("COM3")

        assert transport.port == "COM3"
        assert transport.baudrate == SEYMOUR_BAUD_RATE
        assert transport.reader is None
        assert transport.writer is None

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful serial connection."""
        transport = SerialTransport("/dev/ttyUSB0", 9600)

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_writer = AsyncMock(spec=asyncio.StreamWriter)

        mock_serial_asyncio = Mock()
        mock_serial_asyncio.open_serial_connection = AsyncMock(
            return_value=(mock_reader, mock_writer)
        )

        with patch.dict("sys.modules", {"serial_asyncio": mock_serial_asyncio}):
            await transport.connect()

            mock_serial_asyncio.open_serial_connection.assert_called_once_with(
                url="/dev/ttyUSB0", baudrate=9600
            )
            assert transport.reader is mock_reader
            assert transport.writer is mock_writer

    @pytest.mark.asyncio
    async def test_connect_missing_serial_asyncio(self) -> None:
        """Test connection failure when serial_asyncio is not available."""
        transport = SerialTransport("/dev/ttyUSB0")

        # Simulate import error by removing the module
        with patch.dict("sys.modules", {"serial_asyncio": None}):
            with patch("builtins.__import__") as mock_import:
                mock_import.side_effect = ImportError("No module named 'serial_asyncio'")

                with pytest.raises(SeymourTransportError, match="serial_asyncio not installed"):
                    await transport.connect()

        assert transport.reader is None
        assert transport.writer is None
        await transport.close()

    @pytest.mark.asyncio
    async def test_connect_serial_exception(self) -> None:
        """Test handling of serial connection exceptions."""
        transport = SerialTransport("/dev/ttyUSB0")

        mock_serial_asyncio = Mock()
        mock_serial_asyncio.open_serial_connection = AsyncMock(
            side_effect=Exception("Serial port error")
        )

        with patch.dict("sys.modules", {"serial_asyncio": mock_serial_asyncio}):
            # The open_serial_connection exception is not wrapped, so it propagates directly
            with pytest.raises(Exception, match="Serial port error"):
                await transport.connect()

        assert transport.reader is None
        assert transport.writer is None
        await transport.close()

    @pytest.mark.asyncio
    async def test_full_serial_workflow(self) -> None:
        """Test complete serial transport workflow."""
        transport = SerialTransport("COM3", SEYMOUR_BAUD_RATE)

        mock_reader = AsyncMock(spec=asyncio.StreamReader)
        mock_writer = AsyncMock(spec=asyncio.StreamWriter)

        # Configure mock behaviors
        mock_reader.readuntil.return_value = b"[serial_response]"

        mock_serial_asyncio = Mock()
        mock_serial_asyncio.open_serial_connection = AsyncMock(
            return_value=(mock_reader, mock_writer)
        )

        with patch.dict("sys.modules", {"serial_asyncio": mock_serial_asyncio}):
            # Connect
            await transport.connect()

            # Verify connection parameters
            mock_serial_asyncio.open_serial_connection.assert_called_once_with(
                url="COM3", baudrate=SEYMOUR_BAUD_RATE
            )

            # Send data
            await transport.send(b"[serial_request]")
            mock_writer.write.assert_called_once_with(b"[serial_request]")
            mock_writer.drain.assert_called_once()

            # Receive data
            response = await transport.receive()
            assert response == b"[serial_response]"
            mock_reader.readuntil.assert_called_once_with(b"]")

            # Close connection
            await transport.close()
            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_called_once()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_multiple_connects(self) -> None:
        """Test behavior when connect is called multiple times."""
        transport = TCPTransport("localhost", 8080)

        mock_reader1 = AsyncMock(spec=asyncio.StreamReader)
        mock_writer1 = AsyncMock(spec=asyncio.StreamWriter)
        mock_reader2 = AsyncMock(spec=asyncio.StreamReader)
        mock_writer2 = AsyncMock(spec=asyncio.StreamWriter)

        with patch("asyncio.open_connection") as mock_open:
            # First connection
            mock_open.return_value = (mock_reader1, mock_writer1)
            await transport.connect()
            assert transport.reader is mock_reader1
            assert transport.writer is mock_writer1

            # Second connection (should replace the first)
            mock_open.return_value = (mock_reader2, mock_writer2)
            await transport.connect()
            assert transport.reader is mock_reader2
            assert transport.writer is mock_writer2

    @pytest.mark.asyncio
    async def test_send_after_close(self) -> None:
        """Test sending data after connection is closed."""
        transport = MockTransport()
        await transport.connect()
        await transport.close()

        with pytest.raises(SeymourTransportError, match="Not connected"):
            await transport.send(b"[test]")

    @pytest.mark.asyncio
    async def test_receive_after_close(self) -> None:
        """Test receiving data after connection is closed."""
        transport = MockTransport()
        await transport.connect()
        await transport.close()

        with pytest.raises(SeymourTransportError, match="Not connected"):
            await transport.receive()

    @pytest.mark.asyncio
    async def test_empty_frame(self) -> None:
        """Test handling of empty frames."""
        transport = MockTransport()
        await transport.connect()

        # Empty frame should still be validated
        with pytest.raises(AssertionError, match="Frame must start with '\\['"):
            await transport.send(b"")

    @pytest.mark.asyncio
    async def test_minimal_valid_frame(self) -> None:
        """Test minimal valid frame."""
        transport = MockTransport()
        await transport.connect()

        # Minimal valid frame
        await transport.send(b"[]")

        assert transport.writer is not None
        transport.writer.write.assert_called_once_with(b"[]")

    @pytest.mark.asyncio
    async def test_writer_drain_exception(self) -> None:
        """Test handling of writer.drain() exceptions."""
        transport = MockTransport()
        await transport.connect()

        assert transport.writer is not None
        transport.writer.drain.side_effect = OSError("Network error")

        with pytest.raises(OSError, match="Network error"):
            await transport.send(b"[test]")

    @pytest.mark.asyncio
    async def test_concurrent_close_operations(self) -> None:
        """Test concurrent close operations."""
        transport = MockTransport()
        await transport.connect()

        # Multiple close operations should be safe
        await asyncio.gather(transport.close(), transport.close(), transport.close())

        # References should be None after close operations
        assert transport.reader is None
        assert transport.writer is None

    def test_transport_inheritance(self) -> None:
        """Test that concrete transports properly inherit from base class."""
        tcp_transport = TCPTransport("localhost")
        serial_transport = SerialTransport("/dev/ttyUSB0")

        assert isinstance(tcp_transport, SeymourTransport)
        assert isinstance(serial_transport, SeymourTransport)

        # Verify they have the expected methods
        assert hasattr(tcp_transport, "connect")
        assert hasattr(tcp_transport, "send")
        assert hasattr(tcp_transport, "receive")
        assert hasattr(tcp_transport, "close")

        assert hasattr(serial_transport, "connect")
        assert hasattr(serial_transport, "send")
        assert hasattr(serial_transport, "receive")
        assert hasattr(serial_transport, "close")

    @pytest.mark.asyncio
    async def test_invalid_frame_boundary_characters(self) -> None:
        """Test various invalid frame boundary scenarios."""
        transport = MockTransport()
        await transport.connect()

        # Test cases with different invalid boundaries
        invalid_frames = [
            b"test_frame]",  # Missing opening bracket
            b"[test_frame",  # Missing closing bracket
            b"test_frame",  # Missing both brackets
            b"][test_frame][",  # Wrong order
        ]

        for frame in invalid_frames:
            with pytest.raises(AssertionError):
                await transport.send(frame)

        # This should pass validation (double brackets but valid start/end)
        await transport.send(b"[[test_frame]]")
        assert transport.writer is not None
        transport.writer.write.assert_called_with(b"[[test_frame]]")

    @pytest.mark.asyncio
    async def test_send_frame_validation_edge_cases(self) -> None:
        """Test frame validation with various edge cases."""
        transport = MockTransport()
        await transport.connect()

        # Test cases that should pass validation
        valid_frames = [
            b"[]",  # Minimal frame
            b"[a]",  # Single character
            b"[test with spaces]",  # Spaces in frame
            b"[123]",  # Numeric content
            b"[special!@#$%^&*()_+]",  # Special characters
            b"[\x00\x01\x02]",  # Binary content
        ]

        assert transport.writer is not None

        for frame in valid_frames:
            transport.writer.reset_mock()
            await transport.send(frame)
            transport.writer.write.assert_called_once_with(frame)

    @pytest.mark.asyncio
    async def test_receive_edge_cases(self) -> None:
        """Test receive functionality with edge cases."""
        transport = MockTransport()
        await transport.connect()

        assert transport.reader is not None

        # Test receiving minimal frame
        transport.reader.readuntil.return_value = b"]"
        result = await transport.receive()
        assert result == b"]"

        # Test receiving frame with special characters
        special_frame = b"[special\x00\x01\x02]"
        transport.reader.readuntil.return_value = special_frame
        result = await transport.receive()
        assert result == special_frame

    @pytest.mark.asyncio
    async def test_connection_state_tracking(self) -> None:
        """Test that connection state is properly tracked."""
        transport = MockTransport()

        # Initially not connected
        assert transport.reader is None
        assert transport.writer is None

        # After connect
        await transport.connect()
        assert transport.reader is not None
        assert transport.writer is not None
        assert transport.connect_called

        # After close
        await transport.close()
        assert transport.reader is None
        assert transport.writer is None

    @pytest.mark.asyncio
    async def test_error_propagation(self) -> None:
        """Test that errors from underlying streams are properly propagated."""
        transport = MockTransport()
        await transport.connect()

        assert transport.writer is not None
        assert transport.reader is not None

        # Test write error propagation
        transport.writer.write.side_effect = OSError("Write failed")
        with pytest.raises(OSError, match="Write failed"):
            await transport.send(b"[test]")

        # Test read error propagation
        transport.reader.readuntil.side_effect = OSError("Read failed")
        with pytest.raises(OSError, match="Read failed"):
            await transport.receive()

    @pytest.mark.asyncio
    async def test_close_exception_handling(self) -> None:
        """Test that close properly handles exceptions from writer operations."""
        transport = MockTransport()
        await transport.connect()

        original_writer = transport.writer
        assert original_writer is not None

        # Test that close raises when writer.close() fails but still cleans up
        with patch.object(original_writer, "close", side_effect=Exception("Close failed")):
            # This should raise an exception but cleanup should still happen via finally
            with pytest.raises(Exception, match="Close failed"):
                await transport.close()

        # Verify cleanup still occurred despite the exception
        assert transport.reader is None
        assert transport.writer is None
