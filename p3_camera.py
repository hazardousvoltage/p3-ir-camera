"""P3 Thermal Camera Driver.

USB protocol, frame parsing, temperature conversion, and calibration for
P3-series USB thermal cameras.

Device: VID=0x3474, PID=0x45C2, 160×120 native resolution.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import IntEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray

_ArrT = TypeVar("_ArrT", bound="NDArray[Any]")

# Device constants
VID = 0x3474
HEADER_SIZE = 12
# Temperature conversion constants
TEMP_SCALE = 64  # Raw values are in 1/64 Kelvin units
KELVIN_OFFSET = 273.15


class Model(str, Enum):
    """Camera model."""

    P1 = "p1"  # 160×120 resolution, PID=0x45C2
    P3 = "p3"  # 256×192 resolution, PID=0x45A2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model-specific configuration."""

    model: Model
    pid: int
    frame_w: int
    frame_h: int
    ir_row_end: int
    thermal_row_start: int
    thermal_row_end: int

    @property
    def thermal_rows(self) -> int:
        """Number of thermal rows."""
        return self.thermal_row_end - self.thermal_row_start


def get_model_config(model: Model | str = Model.P3) -> ModelConfig:
    """Get configuration for a camera model.

    Args:
        model: Camera model (P1 or P3).

    Returns:
        Model configuration.
    """
    model = Model(model.lower())
    if model == Model.P1:
        return ModelConfig(
            model=Model.P1,
            pid=0x45C2,
            frame_w=160,
            frame_h=240,  # (120 IR + 2 info + 118 temp) = 240 rows
            ir_row_end=120,
            thermal_row_start=122,
            thermal_row_end=240,
        )
    else:  # P3
        return ModelConfig(
            model=Model.P3,
            pid=0x45A2,
            frame_w=256,
            frame_h=384,  # (192 IR + 2 info + 190 temp) = 384 rows
            ir_row_end=192,
            thermal_row_start=194,
            thermal_row_end=384,
        )


# Default model config (P3 for backward compatibility)
_DEFAULT_CONFIG = get_model_config(Model.P3)
FRAME_W = _DEFAULT_CONFIG.frame_w
FRAME_H = _DEFAULT_CONFIG.frame_h
IR_ROW_START = 0
IR_ROW_END = _DEFAULT_CONFIG.ir_row_end
THERMAL_ROW_START = _DEFAULT_CONFIG.thermal_row_start
THERMAL_ROW_END = _DEFAULT_CONFIG.thermal_row_end
THERMAL_ROWS = _DEFAULT_CONFIG.thermal_rows


class GainMode(IntEnum):
    """Sensor gain mode."""

    LOW = 0  # Extended range: 0°C to 550°C, lower sensitivity
    HIGH = 1  # Limited range: -20°C to 150°C, higher sensitivity
    AUTO = 2  # Auto-switching between HIGH and LOW


@dataclass(slots=True)
class EnvParams:
    """Environmental parameters for temperature correction."""

    emissivity: float = 0.95  # Surface emissivity (0.0-1.0)
    ambient_temp: float = 25.0  # Ambient temperature (°C)
    reflected_temp: float = 25.0  # Reflected/target temperature (°C)
    distance: float = 1.0  # Distance to target (meters, 0.25-49.99)
    humidity: float = 0.5  # Relative humidity (0.0-1.0)


# Pre-computed USB commands with CRC
COMMANDS: dict[str, bytes] = {
    "read_name": bytes.fromhex("0101810001000000000000001e0000004f90"),
    "read_version": bytes.fromhex("0101810002000000000000000c0000001f63"),
    "read_model": bytes.fromhex("010181000600000000000000400000004f65"),
    "read_serial": bytes.fromhex("01018100070000000000000040000000104c"),
    "status": bytes.fromhex("102181000000000000000000020000009501"),
    "start_stream": bytes.fromhex("012f8100000000000000000001000000493f"),
    "gain_low": bytes.fromhex("012f41000000000000000000000000003c3a"),
    "gain_high": bytes.fromhex("012f4100010000000000000000000000493f"),
    "shutter": bytes.fromhex("01364300000000000000000000000000cd0b"),
}


# =============================================================================
# Temperature Conversion (Pure Functions)
# =============================================================================


def raw_to_kelvin(raw: float | NDArray[np.uint16]) -> float | NDArray[np.float32]:
    """Convert raw sensor value to Kelvin.

    Raw values are in 1/64 Kelvin units (centikelvin).

    Args:
        raw: Raw 16-bit sensor value(s).

    Returns:
        Temperature in Kelvin.

    """
    return np.float32(raw) / TEMP_SCALE


def kelvin_to_celsius(
    kelvin: float | NDArray[np.float32],
) -> float | NDArray[np.float32]:
    """Convert Kelvin to Celsius.

    Args:
        kelvin: Temperature in Kelvin.

    Returns:
        Temperature in Celsius.

    """
    return kelvin - KELVIN_OFFSET


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Temperature in Kelvin.

    """
    return celsius + KELVIN_OFFSET


def raw_to_celsius(raw: float | NDArray[np.uint16]) -> float | NDArray[np.float32]:
    """Convert raw sensor value directly to Celsius.

    Formula: (raw / 64) - 273.15

    Args:
        raw: Raw 16-bit sensor value(s).

    Returns:
        Temperature in Celsius.

    """
    return kelvin_to_celsius(raw_to_kelvin(raw))


def celsius_to_raw(celsius: float) -> int:
    """Convert Celsius to raw sensor value.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Raw 16-bit sensor value.

    """
    return int((celsius + KELVIN_OFFSET) * TEMP_SCALE)


# =============================================================================
# Frame Parsing (Pure Functions)
# =============================================================================


def _fix_alignment_quirk(img: _ArrT) -> _ArrT:
    """Fix column alignment quirk in P-series camera data.

    The P3 camera has a hardware quirk where the first 12 columns of each row
    are transmitted at the end of the previous row's USB data. This appears to
    be a sensor readout timing issue: the sensor starts reading at column 12,
    wraps to columns 0-11, but USB transfer packs data linearly.

    This causes:
    1. Columns 0-11 to appear shifted up by one row relative to columns 12-159 (P1) or 12-255 (P3)
    2. The last row of columns 0-11 to contain garbage (no source data)

    Fix:
    1. Roll image left by 12 columns (moves cols 0-11 to cols 148-159 for P1 or 244-255 for P3)
    2. Roll those 12 columns up by 1 row to realign with the rest
    3. Copy second-to-last row over last row to hide garbage

    The 12-column (24-byte) offset suggests DMA/USB buffer alignment boundary.
    Note: This quirk may be specific to P3; P1 may not require this fix.
    """
    col_offset = -12
    row_shift = -1

    result = np.roll(img, col_offset, axis=1)
    n_edge = abs(col_offset)
    result[:, -n_edge:] = np.roll(result[:, -n_edge:], row_shift, axis=0)
    result[-1, -n_edge:] = result[-2, -n_edge:]
    return result


def extract_thermal_data(
    frame_data: bytes,
    apply_col_offset: bool = True,
    config: ModelConfig | None = None,
) -> NDArray[np.uint16] | None:
    """Extract temperature image from raw frame data.

    Args:
        frame_data: Raw USB frame data.
        apply_col_offset: Whether to apply column alignment fix.
        config: Model configuration (defaults to P3).

    Returns:
        Temperature image as uint16 array, or None if invalid.

    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = HEADER_SIZE + config.frame_w * config.frame_h * 2
    if len(frame_data) < expected_size:
        return None

    pixels = np.frombuffer(
        frame_data[HEADER_SIZE : HEADER_SIZE + config.frame_w * config.frame_h * 2],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_h, config.frame_w))

    # Extract thermal region
    thermal = full_frame[config.thermal_row_start : config.thermal_row_end, :].copy()

    if apply_col_offset:
        thermal = _fix_alignment_quirk(thermal)

    return thermal


def extract_ir_brightness(
    frame_data: bytes,
    apply_col_offset: bool = True,
    config: ModelConfig | None = None,
) -> NDArray[np.uint8] | None:
    """Extract IR brightness image from raw frame data.

    Rows 0-(ir_row_end-1) contain display data, taking the low byte of each 16-bit
    value. This data is hardware AGC'd by the camera.

    Args:
        frame_data: Raw USB frame data.
        apply_col_offset: Whether to apply column alignment fix.
        config: Model configuration (defaults to P3).

    Returns:
        IR brightness image as uint8 array, or None if invalid.

    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = HEADER_SIZE + config.frame_w * config.frame_h * 2
    if len(frame_data) < expected_size:
        return None

    pixels = np.frombuffer(
        frame_data[HEADER_SIZE : HEADER_SIZE + config.frame_w * config.frame_h * 2],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_h, config.frame_w))

    # Extract IR brightness region
    ir_16bit = full_frame[IR_ROW_START : config.ir_row_end, :].copy()

    # Low byte contains 8-bit brightness
    ir_8bit = (ir_16bit & 0xFF).astype(np.uint8)

    if apply_col_offset:
        ir_8bit = _fix_alignment_quirk(ir_8bit)

    return ir_8bit


def extract_both(
    frame_data: bytes,
    apply_col_offset: bool = True,
    config: ModelConfig | None = None,
) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint16] | None]:
    """Extract both IR brightness and temperature data from frame.

    Args:
        frame_data: Raw USB frame data.
        apply_col_offset: Whether to apply column alignment fix.
        config: Model configuration (defaults to P3).

    Returns:
        Tuple of (ir_brightness, temperature), either can be None on error.

    """
    if config is None:
        config = _DEFAULT_CONFIG
    expected_size = HEADER_SIZE + config.frame_w * config.frame_h * 2
    if len(frame_data) < expected_size:
        return None, None

    pixels = np.frombuffer(
        frame_data[HEADER_SIZE : HEADER_SIZE + config.frame_w * config.frame_h * 2],
        dtype="<u2",
    )
    full_frame = pixels.reshape((config.frame_h, config.frame_w))

    # IR brightness (low byte)
    ir_16bit = full_frame[IR_ROW_START : config.ir_row_end, :].copy()
    ir_8bit = (ir_16bit & 0xFF).astype(np.uint8)

    # Temperature
    thermal = full_frame[config.thermal_row_start : config.thermal_row_end, :].copy()

    if apply_col_offset:
        ir_8bit = _fix_alignment_quirk(ir_8bit)
        thermal = _fix_alignment_quirk(thermal)

    return ir_8bit, thermal


# =============================================================================
# USB Protocol (Pure Functions)
# =============================================================================


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0x0000) -> int:
    """Compute CRC16-CCITT checksum.

    Args:
        data: Input bytes.
        poly: Polynomial (default: 0x1021 for CCITT).
        init: Initial value.

    Returns:
        16-bit CRC value.

    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def build_command(
    cmd_type: int,
    param: int,
    register: int,
    resp_len: int,
) -> bytes:
    """Build an 18-byte USB command with CRC.

    Command format:
    - Bytes 0-1: Command type (LE)
    - Bytes 2-3: Parameter (LE)
    - Bytes 4-5: Register ID (LE)
    - Bytes 6-13: Reserved (zeros)
    - Bytes 14-15: Response length (LE)
    - Bytes 16-17: CRC16 (LE)

    Args:
        cmd_type: Command type (e.g., 0x0101 for read register).
        param: Parameter value (usually 0x0081).
        register: Register ID.
        resp_len: Expected response length.

    Returns:
        18-byte command with CRC.

    """
    payload = struct.pack(
        "<HHHQH",
        cmd_type,
        param,
        register,
        0,  # 8 bytes reserved
        resp_len,
    )
    crc = crc16_ccitt(payload)
    return payload + struct.pack("<H", crc)


# =============================================================================
# Camera Class (Stateful)
# =============================================================================


@dataclass(slots=True)
class P3Camera:
    """P3 Thermal Camera interface.

    Handles USB communication, streaming, and device state.
    """

    dev: Any = None  # usb.core.Device
    streaming: bool = False
    gain_mode: GainMode = GainMode.HIGH
    env_params: EnvParams = field(default_factory=EnvParams)
    config: ModelConfig = field(default_factory=lambda: _DEFAULT_CONFIG)

    def connect(self) -> None:
        """Connect to the camera."""
        import usb.core  # type: ignore[import-untyped]
        import usb.util  # type: ignore[import-untyped]

        self.dev = usb.core.find(idVendor=VID, idProduct=self.config.pid)
        if self.dev is None:
            model_name = self.config.model.value.upper()
            raise RuntimeError(f"{model_name} camera not found (PID=0x{self.config.pid:04X})")
        self._detach_kernel_drivers()
        self._claim_interfaces()

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self.streaming:
            self.stop_streaming()
        self.dev = None

    def init(self) -> tuple[str, str]:
        """Initialize camera and read device info.

        Returns:
            Tuple of (device_name, firmware_version).

        """
        if self.dev is None:
            raise RuntimeError("Not connected")

        self._send_command(COMMANDS["read_name"])
        time.sleep(0.02)
        name = bytes(self._read_response(30))
        name_str = name.rstrip(b"\x00").decode(errors="replace")

        self._send_command(COMMANDS["read_version"])
        time.sleep(0.02)
        version = bytes(self._read_response(12))
        version_str = version.rstrip(b"\x00").decode(errors="replace")

        self._send_command(COMMANDS["status"])
        self._read_status()

        return name_str, version_str

    def start_streaming(self) -> None:
        """Start video streaming."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        self.dev.set_interface_altsetting(interface=1, alternate_setting=1)
        self.dev.ctrl_transfer(0x40, 0xEE, 0, 1, None, 1000)
        self._send_command(COMMANDS["start_stream"])
        time.sleep(0.1)
        self.streaming = True

    def stop_streaming(self) -> None:
        """Stop video streaming."""
        if self.streaming and self.dev is not None:
            self.dev.set_interface_altsetting(interface=1, alternate_setting=0)
            self.streaming = False

    def read_frame_both(
        self,
    ) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint16] | None]:
        """Read both IR brightness and temperature data.

        Returns:
            Tuple of (ir_brightness uint8, temperature uint16).
            Either can be None on error.

        """
        if self.dev is None or not self.streaming:
            return None, None

        raw_data = self._read_raw_frame()
        return extract_both(raw_data, config=self.config)

    def trigger_shutter(self) -> None:
        """Trigger shutter/NUC calibration."""
        self._send_command(COMMANDS["shutter"])

    def set_gain_mode(self, mode: GainMode) -> None:
        """Set sensor gain mode.

        Args:
            mode: Gain mode (LOW, HIGH, or AUTO).

        """
        if mode == GainMode.LOW:
            self._send_command(COMMANDS["gain_low"])
        elif mode == GainMode.HIGH:
            self._send_command(COMMANDS["gain_high"])
        # AUTO mode requires firmware support
        self.gain_mode = mode

    # Private methods

    def _send_command(self, cmd: bytes) -> None:
        """Send a control command to the device."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        self.dev.ctrl_transfer(0x41, 0x20, 0, 0, cmd, 1000)

    def _read_response(self, length: int) -> bytes:
        """Read response data from the device."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        return bytes(self.dev.ctrl_transfer(0xC1, 0x21, 0, 0, length, 1000))

    def _read_status(self) -> int:
        """Read status byte from the device."""
        if self.dev is None:
            raise RuntimeError("Not connected")
        return self.dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1, 1000)[0]

    def _read_raw_frame(self) -> bytes:
        """Read raw frame data from bulk endpoint."""
        if self.dev is None:
            return b""
        data = b""
        target = HEADER_SIZE + self.config.frame_w * self.config.frame_h * 2
        while len(data) < target:
            try:
                chunk = self.dev.read(0x81, 16384, 1000)
                data += bytes(chunk)
            except Exception:
                break
        return data

    def _detach_kernel_drivers(self) -> None:
        """Detach kernel drivers from USB interfaces."""
        if self.dev is None:
            return
        for cfg in self.dev:
            for intf in cfg:
                try:
                    if self.dev.is_kernel_driver_active(intf.bInterfaceNumber):
                        self.dev.detach_kernel_driver(intf.bInterfaceNumber)
                except Exception:
                    pass

    def _claim_interfaces(self) -> None:
        """Claim USB interfaces."""
        import usb.util  # type: ignore[import-untyped]

        if self.dev is None:
            return
        self.dev.set_configuration()
        usb.util.claim_interface(self.dev, 0)
        usb.util.claim_interface(self.dev, 1)
