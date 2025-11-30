"""
Implementation for the Seymour RS232 protocol version 01.
"""

from __future__ import annotations

import enum
import re

import attrs

FRAME_START = b"["
FRAME_END = b"]"
PROTOCOL_VERSION = b"01"


class SingleByte(enum.StrEnum):
    """
    Base class for enums that represent a 1-byte code.
    """

    @classmethod
    def regex(cls, n: int | None = None) -> bytes:
        """
        Regular expression that parses exactly one code value into a capture group named after the class.
        """
        all_commands = "".join(code.value for code in cls)
        capture_group_suffix = str(n) if n is not None else ""
        return rf"(?P<{cls.__name__}{capture_group_suffix}>[{all_commands}])".encode("ascii")


class CommandCode(SingleByte):
    MOVE_OUT = "O"
    MOVE_IN = "I"
    MOVE_TO_RATIO = "M"
    HOME = "A"
    HALT = "H"
    CALIBRATE = "C"
    STATUS = "S"
    POSITIONS = "P"
    UPDATE_RATIO = "U"
    READ_SYSTEM_INFO = "Y"
    READ_SETTINGS = "R"
    CLEAR_SETTINGS = "X"
    DIAGNOSTICS = "@"


class MotorID(SingleByte):
    TOP = "T"
    BOTTOM = "B"
    LEFT = "L"
    RIGHT = "R"
    VERTICAL = "V"
    HORIZONTAL = "H"
    ALL = "A"


class MovementCode(enum.StrEnum):
    UNTIL_LIMIT = ""
    JOG = "J"  # smallest increment (<0.1%)
    MOVE = "M"  # 1% of the motor's range (ignored if not calibrated)


class StatusCode(SingleByte):
    STOPPED_AT_RATIO = "P"
    MOVING_TO_RATIO = "M"
    HALTED = "H"
    HOMING = "A"
    CALIBRATING = "C"
    MOVING_OUTWARD = "O"
    MOVING_INWARD = "I"
    ERROR = "E"


@attrs.frozen
class Ratio:
    id: str = attrs.field()

    @id.validator
    def check(self, _attribute: attrs.Attribute[str], value: str) -> None:
        if len(value) != 3:
            raise ValueError("ratio_id must be three digits long")
        if not value.isdigit():
            raise ValueError("ratio_id must be numeric")

    @classmethod
    def regex(cls) -> bytes:
        """
        Regular expression that parses a ratio ID into a capture group named 'Ratio'.
        """
        return rb"(?P<Ratio>[0-9]{3})"


@attrs.frozen
class MaskStatus:
    code: StatusCode
    ratio: Ratio | None = None


@attrs.frozen
class MaskPosition:
    motor_id: MotorID
    position_pct: float


@attrs.frozen
class Serial:
    model_code: str
    month: int
    year: int
    production_number: str

    def to_serial_number(self) -> str:
        return f"{self.model_code}-{self.month:02d}{self.year:02d}-{self.production_number}"

    PARSER_RE = re.compile(
        rb"(?P<model_code>.{2})-(?P<month>[0-9]{2})(?P<year>[0-9]{2})-(?P<production_number>.{5})"
    )

    @staticmethod
    def from_serial_number(serial_number: bytes) -> Serial:
        """
        Parse a serial number bytestring into a Serial object.

        The expected format is "XX-MMYY-PPPPP", where XX is the model code and PPPPP is the
        production number. The hyphens are required.
        """
        match = Serial.PARSER_RE.fullmatch(serial_number)
        if not match:
            raise ValueError(f"Malformed serial number: {serial_number!r}")

        model_code = match.group("model_code").decode("ascii")
        month = int(match.group("month"))
        year = int(match.group("year"))
        production_number = match.group("production_number").decode("ascii")
        return Serial(
            model_code=model_code,
            month=month,
            year=year,
            production_number=production_number,
        )


@attrs.frozen
class SystemInfo:
    screen_model: str
    width_inches: float
    height_inches: float
    serial_number: Serial
    mask_ids: list[MotorID]


@attrs.frozen
class RatioSetting:
    ratio: Ratio
    label: str
    width_inches: float
    height_inches: float
    motor_positions_pct: list[float]
    motor_adjustments_pct: list[float]

    @staticmethod
    def from_response_entry(entry: bytes, num_motors: int) -> RatioSetting:
        """
        Parse a ratio setting info string into a RatioSetting object.

        The expected format is:
        - 3 bytes: ratio ID
        - 8 bytes: label (ASCII, space-padded)
        - 6 bytes: width in inches (ASCII float)
        - 6 bytes: height in inches (ASCII float)
        - For each motor:
          - 4 bytes: position percentage (ASCII float)
        - For each motor:
          - 4 bytes: position adjustment percentage (ASCII float)
        """
        base_pattern = Ratio.regex() + rb"(?P<label>.{8})(?P<width>[0-9.]{6})(?P<height>[0-9.]{6})"

        pattern = base_pattern + _ASCII_PCT_RE * num_motors * 2

        match = re.fullmatch(pattern, entry)
        if not match:
            raise ValueError(f"Malformed ratio setting entry for pattern {pattern!r}: {entry!r}")

        ratio_id = match.group("Ratio").decode("ascii")
        label = match.group("label").decode("ascii").strip()
        width = float(match.group("width"))
        height = float(match.group("height"))

        # parse the motor positions and adjustments
        motor_positions_pct: list[float] = []
        motor_adjustments_pct: list[float] = []
        position_group_start = 5  # after ratio, label, width, height
        for i in range(num_motors):
            pos_str = match.group(position_group_start + i)
            adj_str = match.group(position_group_start + num_motors + i)
            motor_positions_pct.append(float(pos_str))
            motor_adjustments_pct.append(float(adj_str))

        return RatioSetting(
            ratio=Ratio(ratio_id),
            label=label,
            width_inches=width,
            height_inches=height,
            motor_positions_pct=motor_positions_pct,
            motor_adjustments_pct=motor_adjustments_pct,
        )

    @staticmethod
    def expected_entry_length(num_motors: int) -> int:
        """
        Calculate the expected length of a ratio setting entry for the given number of motors.
        """
        # 3 (ratio ID) + 8 (label) + 6 (width) + 6 (height) + N * 4 (positions) + N * 4 (adjustments)
        return 3 + 8 + 6 + 6 + num_motors * 4 + num_motors * 4


class DiagnosticCommand(SingleByte):
    DEBUG_LOG = "D"


class DiagnosticOption(enum.StrEnum):
    LIST_FS = "00"
    LIST_SETTINGS_JSON = "10"
    LIST_SYSTEM_JSON = "20"


def _frame(payload: str) -> bytes:
    return FRAME_START + PROTOCOL_VERSION + payload.encode("ascii") + FRAME_END


def encode_move_out(motor_id: MotorID, movement: MovementCode) -> bytes:
    return _frame(f"{CommandCode.MOVE_OUT}{motor_id}{movement}")


def encode_move_in(motor_id: MotorID, movement: MovementCode) -> bytes:
    return _frame(f"{CommandCode.MOVE_IN}{motor_id}{movement}")


def encode_move_ratio(ratio: Ratio) -> bytes:
    return _frame(f"{CommandCode.MOVE_TO_RATIO}{ratio.id}")


def encode_home(motor_id: MotorID) -> bytes:
    return _frame(f"{CommandCode.HOME}{motor_id}")


def encode_halt(motor_id: MotorID) -> bytes:
    return _frame(f"{CommandCode.HALT}{motor_id}")


def encode_calibrate(motor_id: MotorID) -> bytes:
    return _frame(f"{CommandCode.CALIBRATE}{motor_id}")


def encode_status() -> bytes:
    return _frame(CommandCode.STATUS)


def encode_positions() -> bytes:
    return _frame(CommandCode.POSITIONS)


def encode_update_ratio(ratio: Ratio) -> bytes:
    return _frame(f"{CommandCode.UPDATE_RATIO}{ratio.id}")


def encode_clear_settings(ratio: Ratio | None) -> bytes:
    id = ratio.id if ratio is not None else ""
    return _frame(f"{CommandCode.CLEAR_SETTINGS}{id}")


def encode_read_sysinfo() -> bytes:
    return _frame(CommandCode.READ_SYSTEM_INFO)


def encode_read_settings() -> bytes:
    return _frame(CommandCode.READ_SETTINGS)


def encode_diagnostics(option: DiagnosticOption) -> bytes:
    return _frame(f"{CommandCode.DIAGNOSTICS}{DiagnosticCommand.DEBUG_LOG}{option}")


def _frame_re(regex_body: bytes) -> re.Pattern[bytes]:
    regex = b"^\\" + FRAME_START + PROTOCOL_VERSION + regex_body + b"\\" + FRAME_END + b"$"
    return re.compile(regex)


_STATUS_RE = _frame_re(StatusCode.regex() + Ratio.regex() + rb"?")


def decode_status(raw: bytes) -> MaskStatus:
    match = _STATUS_RE.match(raw)
    if not match:
        raise ValueError(f"Malformed status response: {raw!r}")
    status_code = match.group("StatusCode").decode("ascii")
    maybe_ratio_str = match.group("Ratio")
    ratio = Ratio(maybe_ratio_str.decode("ascii")) if maybe_ratio_str is not None else None
    return MaskStatus(code=StatusCode(status_code), ratio=ratio)


_POSITIONS_RE = _frame_re(rb"(?P<num_motors>[1-4])(?P<entries>.+)")
_ASCII_PCT_RE = rb"([0-9.\-]{4})"


def decode_positions(raw: bytes) -> list[MaskPosition]:
    match = _POSITIONS_RE.match(raw)
    if not match:
        raise ValueError(f"Malformed positions response: {raw!r}")
    num_motors = int(match.group("num_motors"))
    entries_str = match.group("entries")

    pattern = b"".join(MotorID.regex(i) + _ASCII_PCT_RE for i in range(num_motors))
    match = re.fullmatch(pattern, entries_str)
    if not match:
        raise ValueError(f"Malformed motor position entries: {entries_str!r}")
    if len(match.groups()) != num_motors * 2:
        raise ValueError(f"Expected {num_motors} motor entries, got: {entries_str!r}")
    positions: list[MaskPosition] = []
    for i in range(num_motors):
        motor_id_str = match.group(1 + i * 2).decode("ascii")
        position_str = match.group(2 + i * 2).decode("ascii")
        positions.append(
            MaskPosition(
                motor_id=MotorID(motor_id_str),
                position_pct=float(position_str),
            )
        )
    return positions


_SYSTEM_INFO_RE = _frame_re(
    rb"(?P<model>.{20})(?P<width>[0-9.]{6})(?P<height>[0-9.]{6})(?P<serial>.{13})(?P<mask_ids>[LRTB]{1,5})"
)


def decode_system_info(raw: bytes) -> SystemInfo:
    match = _SYSTEM_INFO_RE.match(raw)
    if not match:
        raise ValueError(f"Malformed system info response: {raw!r}")
    model = match.group("model").decode("ascii").strip()

    width = float(match.group("width"))
    height = float(match.group("height"))

    serial_str = match.group("serial")
    serial = Serial.from_serial_number(serial_str)

    mask_ids_str = match.group("mask_ids").decode("ascii")
    mask_ids = [MotorID(c) for c in mask_ids_str]

    return SystemInfo(
        screen_model=model,
        width_inches=width,
        height_inches=height,
        serial_number=serial,
        mask_ids=mask_ids,
    )


_SETTINGS_RE = _frame_re(rb"(?P<num_motors>[0-9])(?P<num_ratios>[0-9]{2})(?P<entries>.+)")


def decode_settings(raw: bytes) -> list[RatioSetting]:
    match = _SETTINGS_RE.match(raw)
    if not match:
        raise ValueError(f"Malformed settings response: {raw!r}")
    num_motors = int(match.group("num_motors"))
    num_ratios = int(match.group("num_ratios"))
    entries_str = match.group("entries")

    expected_length_per_entry = RatioSetting.expected_entry_length(num_motors)
    if len(entries_str) != expected_length_per_entry * num_ratios:
        raise ValueError(
            f"Expected {num_ratios} ratio entries of length {expected_length_per_entry}, "
            f"got total length {len(entries_str)}: {entries_str!r}"
        )

    settings: list[RatioSetting] = []
    for i in range(num_ratios):
        entry_bytes = entries_str[
            i * expected_length_per_entry : (i + 1) * expected_length_per_entry
        ]
        setting = RatioSetting.from_response_entry(entry_bytes, num_motors)
        settings.append(setting)

    return settings
