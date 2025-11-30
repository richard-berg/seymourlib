import re

import pytest

from seymourlib import protocol


class TestSingleByte:
    """Test the SingleByte enum base class and its regex functionality."""

    def test_command_code_regex(self) -> None:
        """Test that CommandCode.regex() produces a valid regex pattern."""
        regex = protocol.CommandCode.regex()
        pattern = re.compile(regex)

        # Test valid commands
        assert pattern.fullmatch(b"O")
        assert pattern.fullmatch(b"I")
        assert pattern.fullmatch(b"S")

        # Test invalid commands
        assert not pattern.match(b"Z")
        assert not pattern.fullmatch(b"")
        assert not pattern.fullmatch(b"OO")

    def test_motor_id_regex(self) -> None:
        """Test that MotorID.regex() produces a valid regex pattern."""
        regex = protocol.MotorID.regex()
        pattern = re.compile(regex)

        # Test valid motor IDs
        assert pattern.fullmatch(b"T")
        assert pattern.fullmatch(b"B")
        assert pattern.fullmatch(b"A")

        # Test invalid motor IDs
        assert not pattern.match(b"Z")
        assert not pattern.match(b"")
        assert not pattern.fullmatch(b"TT")

    def test_status_code_regex(self) -> None:
        """Test that StatusCode.regex() produces a valid regex pattern."""
        regex = protocol.StatusCode.regex()
        pattern = re.compile(regex)

        # Test valid status codes
        assert pattern.fullmatch(b"P")
        assert pattern.fullmatch(b"M")
        assert pattern.fullmatch(b"E")

        # Test invalid status codes
        assert not pattern.match(b"Z")
        assert not pattern.match(b"")


class TestRatio:
    """Test the Ratio class validation and functionality."""

    def test_valid_ratio_creation(self) -> None:
        """Test creating valid Ratio objects."""
        ratio = protocol.Ratio("123")
        assert ratio.id == "123"

        ratio = protocol.Ratio("000")
        assert ratio.id == "000"

        ratio = protocol.Ratio("999")
        assert ratio.id == "999"

    def test_invalid_ratio_length(self) -> None:
        """Test that invalid length ratio IDs are rejected."""
        with pytest.raises(ValueError, match="ratio_id must be three digits long"):
            protocol.Ratio("12")

        with pytest.raises(ValueError, match="ratio_id must be three digits long"):
            protocol.Ratio("1234")

        with pytest.raises(ValueError, match="ratio_id must be three digits long"):
            protocol.Ratio("")

    def test_invalid_ratio_non_numeric(self) -> None:
        """Test that non-numeric ratio IDs are rejected."""
        with pytest.raises(ValueError, match="ratio_id must be numeric"):
            protocol.Ratio("abc")

        with pytest.raises(ValueError, match="ratio_id must be numeric"):
            protocol.Ratio("12a")

        with pytest.raises(ValueError, match="ratio_id must be numeric"):
            protocol.Ratio("1.2")

    def test_ratio_regex(self) -> None:
        """Test the Ratio regex pattern."""
        regex = protocol.Ratio.regex()
        pattern = re.compile(regex)

        # Test valid ratios
        match = pattern.fullmatch(b"123")
        assert match
        assert match.group("Ratio") == b"123"

        match = pattern.fullmatch(b"000")
        assert match
        assert match.group("Ratio") == b"000"

        # Test invalid ratios
        assert not pattern.match(b"12")
        assert not pattern.fullmatch(b"1234")
        assert not pattern.match(b"abc")


class TestSerial:
    """Test the Serial class parsing and formatting."""

    def test_valid_serial_parsing(self) -> None:
        """Test parsing valid serial numbers."""
        serial = protocol.Serial.from_serial_number(b"AB-1225-12345")
        assert serial.model_code == "AB"
        assert serial.month == 12
        assert serial.year == 25
        assert serial.production_number == "12345"

    def test_serial_to_string(self) -> None:
        """Test converting Serial object to string."""
        serial = protocol.Serial("AB", 12, 25, "12345")
        assert serial.to_serial_number() == "AB-1225-12345"

    def test_serial_round_trip(self) -> None:
        """Test parsing and formatting round trip."""
        original = b"XY-0199-ABCDE"
        serial = protocol.Serial.from_serial_number(original)
        formatted = serial.to_serial_number().encode()
        assert formatted == original

    def test_invalid_serial_format(self) -> None:
        """Test that malformed serial numbers are rejected."""
        with pytest.raises(ValueError, match="Malformed serial number"):
            protocol.Serial.from_serial_number(b"AB1225-12345")  # Missing hyphen

        with pytest.raises(ValueError, match="Malformed serial number"):
            protocol.Serial.from_serial_number(b"AB-12-12345")  # Wrong month/year format

        with pytest.raises(ValueError, match="Malformed serial number"):
            protocol.Serial.from_serial_number(b"ABC-1225-12345")  # Wrong model code length

        with pytest.raises(ValueError, match="Malformed serial number"):
            protocol.Serial.from_serial_number(b"AB-1225-1234")  # Wrong production number length


class TestRatioSetting:
    """Test the RatioSetting class parsing and functionality."""

    def test_expected_entry_length(self) -> None:
        """Test calculation of expected entry length."""
        # 3 (ratio) + 8 (label) + 6 (width) + 6 (height) + N*4 (positions) + N*4 (adjustments)
        assert protocol.RatioSetting.expected_entry_length(1) == 3 + 8 + 6 + 6 + 4 + 4  # 31
        assert protocol.RatioSetting.expected_entry_length(2) == 3 + 8 + 6 + 6 + 8 + 8  # 39
        assert protocol.RatioSetting.expected_entry_length(4) == 3 + 8 + 6 + 6 + 16 + 16  # 55

    def test_valid_ratio_setting_parsing_1_motor(self) -> None:
        """Test parsing a valid ratio setting with 1 motor."""
        # Format: ratio(3) + label(8) + width(6) + height(6) + position(4) + adjustment(4)
        entry = b"123TestLbl11234.04321.050.0-5.0"
        setting = protocol.RatioSetting.from_response_entry(entry, 1)

        assert setting.ratio.id == "123"
        assert setting.label == "TestLbl1"
        assert setting.width_inches == 1234.0
        assert setting.height_inches == 4321.0
        assert setting.motor_positions_pct == [50.0]
        assert setting.motor_adjustments_pct == [-5.0]

    def test_valid_ratio_setting_parsing_2_motors(self) -> None:
        """Test parsing a valid ratio setting with 2 motors."""
        # Format: ratio(3) + label(8) + width(6) + height(6) + positions(8) + adjustments(8)
        entry = b"456Test2   20.0000015.025.075.0-2.03.00"
        setting = protocol.RatioSetting.from_response_entry(entry, 2)

        assert setting.ratio.id == "456"
        assert setting.label == "Test2"
        assert setting.width_inches == 20.0
        assert setting.height_inches == 15.0
        assert setting.motor_positions_pct == [25.0, 75.0]
        assert setting.motor_adjustments_pct == [-2.0, 3.0]

    def test_invalid_ratio_setting_format(self) -> None:
        """Test that malformed ratio setting entries are rejected."""
        with pytest.raises(ValueError, match="Malformed ratio setting entry"):
            # Too short
            protocol.RatioSetting.from_response_entry(b"123Test", 1)

        with pytest.raises(ValueError, match="Malformed ratio setting entry"):
            # Invalid ratio format
            protocol.RatioSetting.from_response_entry(b"abcTestLbl16.0  12.0  50.0-5.0", 1)


class TestEncoding:
    """Test all encoding functions."""

    def test_encode_status_frame(self) -> None:
        """Test encoding status request."""
        result = protocol.encode_status()
        assert result == b"[01S]"

    def test_encode_positions_frame(self) -> None:
        """Test encoding positions request."""
        result = protocol.encode_positions()
        assert result == b"[01P]"

    def test_encode_read_sysinfo(self) -> None:
        """Test encoding system info request."""
        result = protocol.encode_read_sysinfo()
        assert result == b"[01Y]"

    def test_encode_move_out(self) -> None:
        """Test encoding move out commands."""
        result = protocol.encode_move_out(protocol.MotorID.TOP, protocol.MovementCode.UNTIL_LIMIT)
        assert result == b"[01OT]"

        result = protocol.encode_move_out(protocol.MotorID.LEFT, protocol.MovementCode.JOG)
        assert result == b"[01OLJ]"

        result = protocol.encode_move_out(protocol.MotorID.ALL, protocol.MovementCode.MOVE)
        assert result == b"[01OAM]"

    def test_encode_move_in(self) -> None:
        """Test encoding move in commands."""
        result = protocol.encode_move_in(protocol.MotorID.BOTTOM, protocol.MovementCode.UNTIL_LIMIT)
        assert result == b"[01IB]"

        result = protocol.encode_move_in(protocol.MotorID.RIGHT, protocol.MovementCode.JOG)
        assert result == b"[01IRJ]"

    def test_encode_move_ratio(self) -> None:
        """Test encoding move to ratio commands."""
        ratio = protocol.Ratio("123")
        result = protocol.encode_move_ratio(ratio)
        assert result == b"[01M123]"

        ratio = protocol.Ratio("000")
        result = protocol.encode_move_ratio(ratio)
        assert result == b"[01M000]"

    def test_encode_home(self) -> None:
        """Test encoding home commands."""
        result = protocol.encode_home(protocol.MotorID.TOP)
        assert result == b"[01AT]"

        result = protocol.encode_home(protocol.MotorID.ALL)
        assert result == b"[01AA]"

    def test_encode_halt(self) -> None:
        """Test encoding halt commands."""
        result = protocol.encode_halt(protocol.MotorID.VERTICAL)
        assert result == b"[01HV]"

        result = protocol.encode_halt(protocol.MotorID.ALL)
        assert result == b"[01HA]"

    def test_encode_calibrate(self) -> None:
        """Test encoding calibrate commands."""
        result = protocol.encode_calibrate(protocol.MotorID.HORIZONTAL)
        assert result == b"[01CH]"

        result = protocol.encode_calibrate(protocol.MotorID.ALL)
        assert result == b"[01CA]"

    def test_encode_update_ratio(self) -> None:
        """Test encoding update ratio commands."""
        ratio = protocol.Ratio("456")
        result = protocol.encode_update_ratio(ratio)
        assert result == b"[01U456]"


class TestDecoding:
    """Test all decoding functions."""

    def test_decode_status_with_ratio(self) -> None:
        """Test decoding status responses that include a ratio."""
        status = protocol.decode_status(b"[01P123]")
        assert status.code == protocol.StatusCode.STOPPED_AT_RATIO
        assert status.ratio is not None
        assert status.ratio.id == "123"

    def test_decode_status_without_ratio(self) -> None:
        """Test decoding status responses without a ratio."""
        status = protocol.decode_status(b"[01H]")
        assert status.code == protocol.StatusCode.HALTED
        assert status.ratio is None

        status = protocol.decode_status(b"[01E]")
        assert status.code == protocol.StatusCode.ERROR
        assert status.ratio is None

    def test_decode_status_all_codes(self) -> None:
        """Test decoding all possible status codes."""
        test_cases = [
            (b"[01P123]", protocol.StatusCode.STOPPED_AT_RATIO, "123"),
            (b"[01M456]", protocol.StatusCode.MOVING_TO_RATIO, "456"),
            (b"[01H]", protocol.StatusCode.HALTED, None),
            (b"[01A]", protocol.StatusCode.HOMING, None),
            (b"[01C]", protocol.StatusCode.CALIBRATING, None),
            (b"[01O]", protocol.StatusCode.MOVING_OUTWARD, None),
            (b"[01I]", protocol.StatusCode.MOVING_INWARD, None),
            (b"[01E]", protocol.StatusCode.ERROR, None),
        ]

        for raw_bytes, expected_code, expected_ratio_id in test_cases:
            status = protocol.decode_status(raw_bytes)
            assert status.code == expected_code
            if expected_ratio_id:
                assert status.ratio is not None
                assert status.ratio.id == expected_ratio_id
            else:
                assert status.ratio is None

    def test_decode_positions_single_motor(self) -> None:
        """Test decoding positions response with one motor."""
        positions = protocol.decode_positions(b"[011T50.0]")
        assert len(positions) == 1
        assert positions[0].motor_id == protocol.MotorID.TOP
        assert positions[0].position_pct == 50.0

    def test_decode_positions_multiple_motors(self) -> None:
        """Test decoding positions response with multiple motors."""
        positions = protocol.decode_positions(b"[013T25.5B75.0L-10.]")
        assert len(positions) == 3

        assert positions[0].motor_id == protocol.MotorID.TOP
        assert positions[0].position_pct == 25.5

        assert positions[1].motor_id == protocol.MotorID.BOTTOM
        assert positions[1].position_pct == 75.0

        assert positions[2].motor_id == protocol.MotorID.LEFT
        assert positions[2].position_pct == -10.0

    def test_decode_system_info(self) -> None:
        """Test decoding system info response."""
        raw = b"[01PRH-123             0123.00069.1SS-0325-Berg TB]"
        system_info = protocol.decode_system_info(raw)

        assert system_info.screen_model == "PRH-123"
        assert system_info.width_inches == 123.0
        assert system_info.height_inches == 69.1
        assert system_info.serial_number.model_code == "SS"
        assert system_info.serial_number.month == 3
        assert system_info.serial_number.year == 25
        assert system_info.serial_number.production_number == "Berg "
        assert len(system_info.mask_ids) == 2
        assert system_info.mask_ids[0] == protocol.MotorID.TOP
        assert system_info.mask_ids[1] == protocol.MotorID.BOTTOM

    def test_decode_settings_single_ratio(self) -> None:
        """Test decoding settings response with one ratio."""
        # Format: num_motors(1) + num_ratios(2) + entry(31 bytes for 1 motor)
        raw = b"[01101123TestLbl 160.01092.0050.0-5.0]"
        settings = protocol.decode_settings(raw)

        assert len(settings) == 1
        setting = settings[0]
        assert setting.ratio.id == "123"
        assert setting.label == "TestLbl"
        assert setting.width_inches == 160.01
        assert setting.height_inches == 92.0
        assert setting.motor_positions_pct == [50.0]
        assert setting.motor_adjustments_pct == [-5.0]

    def test_decode_settings_multiple_ratios(self) -> None:
        """Test decoding settings response with multiple ratios."""
        # factory default with 2 motors
        raw = b"[012182402.40    0123.00051.2100.100.00.000.02392.39:1  0123.00051.476.076.000.000.02372.37:1  0123.00051.874.174.100.000.02352.35:1  0123.00052.372.472.400.000.02202.2:1   0123.00055.957.057.000.000.02002:1     0123.00061.532.233.200.000.01851.85:1  0123.00066.412.913.800.000.017816:9    0123.00069.100.000.000.000.0990custom 00123.00000.000.000.000.000.0991custom 10123.00000.000.000.000.000.0992custom 20123.00000.000.000.000.000.0993custom 30123.00000.000.000.000.000.0994custom 40123.00000.000.000.000.000.0995custom 50123.00000.000.000.000.000.0996custom 60123.00000.000.000.000.000.0997custom 70123.00000.000.000.000.000.0998custom 80123.00000.000.000.000.000.0999custom 90123.00000.000.000.000.000.0]"

        settings = protocol.decode_settings(raw)

        # fmt: off
        expected = [
            protocol.RatioSetting(ratio=protocol.Ratio(id='240'), label='2.40', width_inches=123.0, height_inches=51.2, motor_positions_pct=[100.0, 100.0], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='239'), label='2.39:1', width_inches=123.0, height_inches=51.4, motor_positions_pct=[76.0, 76.0], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='237'), label='2.37:1', width_inches=123.0, height_inches=51.8, motor_positions_pct=[74.1, 74.1], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='235'), label='2.35:1', width_inches=123.0, height_inches=52.3, motor_positions_pct=[72.4, 72.4], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='220'), label='2.2:1', width_inches=123.0, height_inches=55.9, motor_positions_pct=[57.0, 57.0], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='200'), label='2:1', width_inches=123.0, height_inches=61.5, motor_positions_pct=[32.2, 33.2], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='185'), label='1.85:1', width_inches=123.0, height_inches=66.4, motor_positions_pct=[12.9, 13.8], motor_adjustments_pct=[0.0, 0.0]),
            protocol.RatioSetting(ratio=protocol.Ratio(id='178'), label='16:9', width_inches=123.0, height_inches=69.1, motor_positions_pct=[0.0, 0.0], motor_adjustments_pct=[0.0, 0.0]),
        ]
        assert settings[:8] == expected

        expected_custom = [
            protocol.RatioSetting(ratio=protocol.Ratio(id=str(990+i)), label=f'custom {i}', width_inches=123.0, height_inches=0.0, motor_positions_pct=[0.0, 0.0], motor_adjustments_pct=[0.0, 0.0])
            for i in range(10)
        ]
        assert settings[8:] == expected_custom
        # fmt: on


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_decode_status_malformed(self) -> None:
        """Test that malformed status responses raise ValueError."""
        with pytest.raises(ValueError, match="Malformed status response"):
            protocol.decode_status(b"[02P123]")  # Wrong protocol version

        with pytest.raises(ValueError, match="Malformed status response"):
            protocol.decode_status(b"[01Z123]")  # Invalid status code

        with pytest.raises(ValueError, match="Malformed status response"):
            protocol.decode_status(b"[01P12]")  # Invalid ratio format

    def test_decode_positions_malformed(self) -> None:
        """Test that malformed positions responses raise ValueError."""
        with pytest.raises(ValueError, match="Malformed positions response"):
            protocol.decode_positions(b"[025T50.0]")  # Wrong protocol version

        with pytest.raises(ValueError, match="Malformed motor position entries"):
            protocol.decode_positions(b"[011Z50.0]")  # Invalid motor ID

        with pytest.raises(ValueError, match="Malformed motor position entries"):
            protocol.decode_positions(b"[012T50.0]")  # Missing second motor

    def test_decode_system_info_malformed(self) -> None:
        """Test that malformed system info responses raise ValueError."""
        with pytest.raises(ValueError, match="Malformed system info response"):
            protocol.decode_system_info(b"[02TestScreen]")  # Wrong protocol version

        with pytest.raises(ValueError, match="Malformed system info response"):
            protocol.decode_system_info(b"[01TooShort]")  # Too short

    def test_decode_settings_malformed(self) -> None:
        """Test that malformed settings responses raise ValueError."""
        with pytest.raises(ValueError, match="Malformed settings response"):
            protocol.decode_settings(b"[02101...]")  # Wrong protocol version

        with pytest.raises(ValueError, match="Expected 1 ratio entries"):
            # Wrong entry length - says 1 ratio but data is too short
            protocol.decode_settings(b"[01101short]")
