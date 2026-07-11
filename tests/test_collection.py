"""Unit tests for some of the collection.py logic, located in src/collection.py"""

# Standard imports
import logging
from unittest.mock import MagicMock, Mock, patch

# Third party imports
import pytest
import serial
from inputimeout import TimeoutOccurred

# Local imports
from ahc.collection import (
    DataCollectionError,
    _validate_audio_data,
    get_arduino_data,
    init_serial_connection,
    input_sample_metadata,
)


@patch("ahc.collection.serial.Serial")
class TestInitSerialConnection:
    """Test cases for the init_serial_connection function."""

    def test_successful_connection(
        self, mock_serial: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test successful serial connection."""
        mock_serial.return_value = MagicMock()
        port = "/dev/ttyACM0"

        with caplog.at_level(logging.INFO):
            ser = init_serial_connection(port)

        mock_serial.assert_called_once_with(port, baudrate=1000000, timeout=10.0)
        assert ser is not None
        assert isinstance(ser, MagicMock)
        assert f"Connected to Arduino on {port} at 1000000 baud." in caplog.text

    def test_failed_connection(self, mock_serial: Mock) -> None:
        """Test failed serial connection."""
        mock_serial.side_effect = serial.SerialException("Connection failed")
        port = "/dev/ttyACM0"

        with pytest.raises(
            DataCollectionError,
            match=f"Failed to connect to Arduino on {port}: Connection failed",
        ):
            init_serial_connection(port)


@patch("ahc.collection.inputimeout")
class TestInputSampleMetadata:
    """Test cases for the input_sample_metadata function."""

    def test_successful_input(self, mock_inputimeout: Mock) -> None:
        """Test successful user input."""
        mock_inputimeout.side_effect = ["table", "hard", "random comment"]
        timeout = 5.0

        metadata = input_sample_metadata(timeout)

        assert isinstance(metadata, dict)
        assert metadata == {
            "material": "table",
            "class": "hard",
            "comment": "random comment",
        }

    def test_timeout_occurred(self, mock_inputimeout: Mock) -> None:
        """Test timeout during user input."""
        mock_inputimeout.side_effect = TimeoutOccurred
        timeout = 5.0

        with pytest.raises(
            DataCollectionError, match="Timeout occurred while waiting for input."
        ):
            input_sample_metadata(timeout)

    def test_wrong_inputs(
        self, mock_inputimeout: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handling of wrong user inputs, and of empty comment input."""
        timeout = 5.0
        mock_inputimeout.side_effect = ["ta", "table", "wrong", "hard", ""]

        with caplog.at_level(logging.INFO):
            metadata = input_sample_metadata(timeout)

        assert metadata == {
            "material": "table",
            "class": "hard",
            "comment": None,
        }
        assert (
            "Wrong input. Please enter a meaningful material name (3+ chars)"
            in caplog.text
        )
        assert "Wrong input. Please enter: hard, medium, or soft" in caplog.text


@patch("ahc.collection._validate_audio_data", side_effect=None)
class TestGetArduinoData:
    """Test cases for the get_arduino_data function."""

    def test_successful_data_retrieval(
        self, mock_validate_audio_data: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test successful data retrieval from Arduino."""
        mock_serial = Mock()
        mock_serial.timeout = 10.0

        # Simulate data available
        mock_serial.in_waiting = 1

        # Simulate chunks arriving over multiple iterations with a valid JSON string
        mock_serial.read.side_effect = [
            b"Recording continuously. Drop object to trigger capture...\n",
            b"--- TRIGGER DETECTED ---\n",
            b"--- RECORDING DATA ---\n",
            b'{"foo": "bar"}\r\n',
        ]

        with caplog.at_level(logging.INFO):
            data = get_arduino_data(mock_serial)

        assert "Listening for trigger..." in caplog.text
        assert (
            "Arduino: Recording continuously. Drop object to trigger capture..."
            in caplog.text
        )
        assert "Arduino: --- TRIGGER DETECTED ---" in caplog.text
        assert "Arduino: --- RECORDING DATA ---" in caplog.text
        assert "Received JSON data from Arduino." in caplog.text
        assert isinstance(data, dict)
        assert data == {"foo": "bar"}

    @patch("ahc.collection.time.perf_counter", side_effect=[0.0, 1.0, 65.0, 125.0])
    def test_timeout_no_data(
        self,
        mock_time: Mock,
        mock_validate_audio_data: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test timeout when no data is received from Arduino."""
        mock_serial = Mock()
        mock_serial.timeout = 10.0
        mock_serial.in_waiting = 0  # Simulate no data available
        fn_timeout = 120.0  # Default timeout for get_arduino_data

        with caplog.at_level(logging.INFO):
            with pytest.raises(
                DataCollectionError, match="Timeout waiting for Arduino data"
            ):
                get_arduino_data(mock_serial, timeout=fn_timeout)

        assert "No data has been received in 60.0s, still listening..." in caplog.text
        assert "Timeout: No data received in 125.0s" in caplog.text

    def test_incomplete_data_received(self, mock_validate_audio_data: Mock) -> None:
        """
        Test incomplete data received from Arduino.
        If function receives "--- END RECORDING ---" before valid JSON data, it means
        the script started in the middle of a recording session (missing `{`).
        """
        mock_serial = Mock()
        mock_serial.timeout = 10.0

        # Simulate data available
        mock_serial.in_waiting = 1

        # Simulate chunks arriving over multiple iterations with a valid JSON string
        mock_serial.read.side_effect = [
            b"--- END RECORDING ---\n",
        ]

        with pytest.raises(
            DataCollectionError,
            match="Unexpected END RECORDING without valid JSON",
        ):
            get_arduino_data(mock_serial)

    def test_invalid_json_received(self, mock_validate_audio_data: Mock) -> None:
        """
        Test invalid JSON data received from Arduino.
        If function receives invalid JSON data, it should raise a DataCollectionError.
        """
        mock_serial = Mock()
        mock_serial.timeout = 10.0

        # Simulate data available
        mock_serial.in_waiting = 1

        # Simulate chunks arriving over multiple iterations with a valid JSON string
        mock_serial.read.side_effect = [
            b"{invalid_json: true}\r\n",
        ]
        with pytest.raises(DataCollectionError, match="Invalid JSON from Arduino"):
            get_arduino_data(mock_serial)

    def test_serial_read_error(self, mock_validate_audio_data: Mock) -> None:
        """Test handling of serial port read errors."""
        mock_serial = Mock()
        mock_serial.timeout = 10.0

        # Simulate data available
        mock_serial.in_waiting = 1

        # Simulate a serial read error
        mock_serial.read.side_effect = Exception("Serial port disconnected")

        with pytest.raises(
            DataCollectionError, match="Serial read error: Serial port disconnected"
        ):
            get_arduino_data(mock_serial)


class TestValidateAudioData:
    """Test cases for the _validate_audio_data function."""

    def test_valid_data(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test valid audio data."""
        data = {
            "arduino_ide_version": "2.3.10",
            "board": "Arduino Nano 33 BLE Sense Rev2",
            "microcontroller": "Nordic nRF52840",
            "microphone": "ST MP34DT06JTR",
            "sample_rate": 16000,
            "duration_seconds": 0.000625,
            "num_samples": 10,
            "pre_trigger_samples": 1,
            "post_trigger_samples": 9,
            "trigger_threshold": 1000,
            "mic_pdm_gain": 20,
            "overflow": 0,
            "values": [4, 1500, -17, 5, 98, -1032, 0, 0, 0, 0],
        }

        with caplog.at_level(logging.INFO):
            _validate_audio_data(data)

        assert "Audio data validated successfully." in caplog.text

    def test_missing_fields(self) -> None:
        """Test data with missing required fields."""
        data = {
            "arduino_ide_version": "2.3.10",
            "values": [4, 1500, -17, 5, 98, -1032, 0, 0, 0, 0],
        }

        with pytest.raises(DataCollectionError, match="Missing required fields"):
            _validate_audio_data(data)

    def test_wrong_sample_count(self) -> None:
        """Test data with wrong sample count."""
        data = {
            "arduino_ide_version": "2.3.10",
            "board": "Arduino Nano 33 BLE Sense Rev2",
            "microcontroller": "Nordic nRF52840",
            "microphone": "ST MP34DT06JTR",
            "sample_rate": 16000,
            "duration_seconds": 2.5,
            "num_samples": 40000,
            "pre_trigger_samples": 800,
            "post_trigger_samples": 39200,
            "trigger_threshold": 1000,
            "mic_pdm_gain": 20,
            "overflow": 0,
            "values": [4, 700, -17, 5, 98, -999, 0, 0, 0, 0],
        }

        with pytest.raises(DataCollectionError, match="Sample count mismatch"):
            _validate_audio_data(data)

    def test_wrong_max_amplitude(self) -> None:
        """Test data with wrong maximum amplitude."""
        data = {
            "arduino_ide_version": "2.3.10",
            "board": "Arduino Nano 33 BLE Sense Rev2",
            "microcontroller": "Nordic nRF52840",
            "microphone": "ST MP34DT06JTR",
            "sample_rate": 16000,
            "duration_seconds": 0.000625,
            "num_samples": 10,
            "pre_trigger_samples": 1,
            "post_trigger_samples": 9,
            "trigger_threshold": 1000,
            "mic_pdm_gain": 20,
            "overflow": 0,
            "values": [4, 700, -17, 5, 98, -999, 0, 0, 0, 0],
        }

        with pytest.raises(DataCollectionError, match="Unexpected max amplitude"):
            _validate_audio_data(data)

    def test_data_with_overflow(self) -> None:
        """Test data with overflow."""
        data = {
            "arduino_ide_version": "2.3.10",
            "board": "Arduino Nano 33 BLE Sense Rev2",
            "microcontroller": "Nordic nRF52840",
            "microphone": "ST MP34DT06JTR",
            "sample_rate": 16000,
            "duration_seconds": 0.000625,
            "num_samples": 10,
            "pre_trigger_samples": 1,
            "post_trigger_samples": 9,
            "trigger_threshold": 1000,
            "mic_pdm_gain": 20,
            "overflow": 1,
            "values": [4, 1500, -17, 5, 98, -1032, 0, 0, 0, 0],
        }

        with pytest.raises(
            DataCollectionError,
            match="Audio buffer overflow detected. Discarding sample.",
        ):
            _validate_audio_data(data)
