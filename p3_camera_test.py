"""Unit tests for p3_camera.py."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from p3_camera import COMMANDS
from p3_camera import FRAME_H
from p3_camera import FRAME_W
from p3_camera import HEADER_SIZE
from p3_camera import KELVIN_OFFSET
from p3_camera import Model
from p3_camera import TEMP_SCALE
from p3_camera import THERMAL_ROW_END
from p3_camera import THERMAL_ROW_START
from p3_camera import THERMAL_ROWS
from p3_camera import EnvParams
from p3_camera import GainMode
from p3_camera import build_command
from p3_camera import celsius_to_kelvin
from p3_camera import celsius_to_raw
from p3_camera import crc16_ccitt
from p3_camera import extract_thermal_data
from p3_camera import get_model_config
from p3_camera import kelvin_to_celsius
from p3_camera import raw_to_celsius
from p3_camera import raw_to_kelvin


class TestConstants:
    """Test that constants have expected values."""

    def test_temp_scale(self):
        assert TEMP_SCALE == 64

    def test_kelvin_offset(self):
        assert KELVIN_OFFSET == 273.15

    def test_frame_dimensions(self):
        # Default is P3
        assert FRAME_W == 256
        assert FRAME_H == 384
        assert THERMAL_ROWS == 190

    def test_thermal_rows(self):
        # Default is P3
        assert THERMAL_ROW_START == 194
        assert THERMAL_ROW_END == 384
        assert THERMAL_ROW_END - THERMAL_ROW_START == THERMAL_ROWS

    def test_p1_model_config(self):
        """Test P1 model configuration."""
        config = get_model_config(Model.P1)
        assert config.model == Model.P1
        assert config.pid == 0x45C2
        assert config.frame_w == 160
        assert config.frame_h == 240
        assert config.ir_row_end == 120
        assert config.thermal_row_start == 122
        assert config.thermal_row_end == 240
        assert config.thermal_rows == 118

    def test_p3_model_config(self):
        """Test P3 model configuration."""
        config = get_model_config(Model.P3)
        assert config.model == Model.P3
        assert config.pid == 0x45A2
        assert config.frame_w == 256
        assert config.frame_h == 384
        assert config.ir_row_end == 192
        assert config.thermal_row_start == 194
        assert config.thermal_row_end == 384
        assert config.thermal_rows == 190


class TestTemperatureConversion:
    """Tests for temperature conversion functions."""

    def test_raw_to_kelvin_zero(self):
        # 0 raw = 0 Kelvin (absolute zero)
        assert raw_to_kelvin(0) == pytest.approx(0.0)

    def test_raw_to_kelvin_room_temp(self):
        # Room temp ~25°C = 298.15K
        # raw = 298.15 * 64 = 19081.6
        raw = int(298.15 * TEMP_SCALE)
        assert raw_to_kelvin(raw) == pytest.approx(298.15, rel=1e-3)

    def test_raw_to_kelvin_array(self):
        raw = np.array([0, 17481, 19082], dtype=np.uint16)
        kelvin = raw_to_kelvin(raw)
        assert isinstance(kelvin, np.ndarray)
        assert kelvin.shape == (3,)
        assert kelvin[0] == pytest.approx(0.0)
        assert kelvin[1] == pytest.approx(273.14, rel=1e-3)  # ~0°C
        assert kelvin[2] == pytest.approx(298.16, rel=1e-3)  # ~25°C

    def test_kelvin_to_celsius_freezing(self):
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0)

    def test_kelvin_to_celsius_boiling(self):
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0)

    def test_kelvin_to_celsius_array(self):
        kelvin = np.array([273.15, 298.15, 373.15], dtype=np.float32)
        celsius = kelvin_to_celsius(kelvin)
        assert isinstance(celsius, np.ndarray)
        assert celsius[0] == pytest.approx(0.0)
        assert celsius[1] == pytest.approx(25.0)
        assert celsius[2] == pytest.approx(100.0)

    def test_raw_to_celsius_freezing(self):
        # 0°C = 273.15K, raw = 273.15 * 64 = 17481.6
        raw = int(273.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(0.0, abs=0.02)

    def test_raw_to_celsius_body_temp(self):
        # 37°C = 310.15K, raw = 310.15 * 64 = 19849.6
        raw = int(310.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(37.0, abs=0.02)

    def test_raw_to_celsius_boiling(self):
        # 100°C = 373.15K, raw = 373.15 * 64 = 23881.6
        raw = int(373.15 * TEMP_SCALE)
        assert raw_to_celsius(raw) == pytest.approx(100.0, abs=0.02)

    def test_celsius_to_raw_roundtrip(self):
        for temp in [-40.0, 0.0, 25.0, 37.0, 100.0, 200.0]:
            raw = celsius_to_raw(temp)
            recovered = raw_to_celsius(raw)
            assert recovered == pytest.approx(temp, abs=0.02)

    def test_celsius_to_kelvin(self):
        assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15)
        assert celsius_to_kelvin(-273.15) == pytest.approx(0.0)


class TestFrameParsing:
    """Tests for frame parsing functions."""

    def test_extract_thermal_data_valid(self):
        # Create synthetic frame data (using P3 default)
        frame_size = HEADER_SIZE + FRAME_W * FRAME_H * 2
        data = bytearray(frame_size)

        # Fill thermal region with recognizable pattern
        pixels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)
        pixels[THERMAL_ROW_START:THERMAL_ROW_END, :] = 20000  # Thermal data

        # Pack into bytes
        data[HEADER_SIZE:] = pixels.tobytes()
        result = extract_thermal_data(bytes(data), apply_col_offset=False)

        assert result is not None
        assert result.shape == (THERMAL_ROWS, FRAME_W)
        assert result[0, 0] == 20000

    def test_extract_thermal_data_p1(self):
        """Test thermal data extraction for P1 model."""
        config = get_model_config(Model.P1)
        frame_size = HEADER_SIZE + config.frame_w * config.frame_h * 2
        data = bytearray(frame_size)

        pixels = np.zeros((config.frame_h, config.frame_w), dtype=np.uint16)
        pixels[config.thermal_row_start : config.thermal_row_end, :] = 20000

        data[HEADER_SIZE:] = pixels.tobytes()
        result = extract_thermal_data(bytes(data), apply_col_offset=False, config=config)

        assert result is not None
        assert result.shape == (config.thermal_rows, config.frame_w)
        assert result[0, 0] == 20000

    def test_extract_thermal_data_too_short(self):
        data = b"\x00" * 100  # Way too short
        result = extract_thermal_data(data)
        assert result is None

    def test_extract_thermal_data_col_offset(self):
        # Test with P3 (default)
        frame_size = HEADER_SIZE + FRAME_W * FRAME_H * 2
        data = bytearray(frame_size)

        # Create pattern with marker at specific column (use middle row to avoid
        # edge effects from _fix_alignment_quirk's row shift on edge columns)
        pixels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)
        pixels[THERMAL_ROW_START + 50, 0] = 12345  # Marker at row 50, col 0

        data[HEADER_SIZE:] = pixels.tobytes()

        # Without offset
        result_no_offset = extract_thermal_data(bytes(data), apply_col_offset=False)
        assert result_no_offset is not None
        assert result_no_offset[50, 0] == 12345

        # With offset (marker should move due to _fix_alignment_quirk)
        result_with_offset = extract_thermal_data(bytes(data), apply_col_offset=True)
        assert result_with_offset is not None
        # Quirk: roll -12 cols (0->244), then roll edge cols up by -1 (50->49)
        assert result_with_offset[49, 244] == 12345


class TestProtocol:
    """Tests for USB protocol functions."""

    def test_crc16_known_values(self):
        payload = bytes.fromhex("0101810001000000000000001e000000")
        crc = crc16_ccitt(payload)
        assert crc == 0x904F

    def test_build_command_length(self):
        cmd = build_command(cmd_type=0x0101, param=0x0081, register=0x0001, resp_len=30)
        assert len(cmd) == 18

    def test_build_command_structure(self):
        cmd = build_command(cmd_type=0x0101, param=0x0081, register=0x0006, resp_len=64)
        # Check command type (first 2 bytes, little-endian)
        assert cmd[0:2] == b"\x01\x01"
        # Check param (bytes 2-3)
        assert cmd[2:4] == b"\x81\x00"
        # Check register (bytes 4-5)
        assert cmd[4:6] == b"\x06\x00"
        # Check response length (bytes 14-15)
        assert cmd[14:16] == b"\x40\x00"  # 64 in little-endian

    def test_precomputed_commands_length(self):
        for name, cmd in COMMANDS.items():
            assert len(cmd) == 18, f"Command {name} has wrong length"


class TestEnums:
    """Tests for enum values."""

    def test_gain_mode_values(self):
        assert GainMode.LOW == 0
        assert GainMode.HIGH == 1
        assert GainMode.AUTO == 2


class TestDataClasses:
    """Tests for dataclass defaults."""

    def test_env_params_defaults(self):
        env = EnvParams()
        assert env.emissivity == 0.95
        assert env.ambient_temp == 25.0
        assert env.distance == 1.0
        assert env.humidity == 0.5


def _run_tests(test_file: str) -> None:
    """Run pytest on this file."""
    sys.exit(
        pytest.main(
            [
                test_file,
                "-v",
                "-s",
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
                *sys.argv[1:],
            ]
        )
    )


if __name__ == "__main__":
    _run_tests(__file__)
