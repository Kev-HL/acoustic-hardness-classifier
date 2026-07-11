"""
Python module for data collection logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import json
import logging
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import serial
from inputimeout import TimeoutOccurred, inputimeout

# Set up logger
logger = logging.getLogger(__name__)


class DataCollectionError(Exception):
    """Custom exception for data collection errors."""

    pass


class InterruptDebouncer:
    """
    Debounce keyboard interrupts on PyInstaller executables.

    PyInstaller's event loop has a higher polling rate for Ctrl+C, causing
    multiple KeyboardInterrupt signals in quick succession.
    """

    def __init__(self, debounce_seconds: float = 0.5):
        self.debounce_seconds = debounce_seconds
        self.last_interrupt_time: float = -float("inf")  # Let first interrupt through
        self.first_interrupt_caught = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, sig, frame):
        current_time = time.perf_counter()

        # First interrupt always goes through
        if not self.first_interrupt_caught:
            self.first_interrupt_caught = True
            self.last_interrupt_time = current_time
            raise KeyboardInterrupt()

        # Next interrupts: ignore if within debounce window
        if current_time - self.last_interrupt_time < self.debounce_seconds:
            return  # Ignore this interrupt

        # If debounce window has passed, allow the interrupt
        self.last_interrupt_time = current_time
        raise KeyboardInterrupt()


def init_serial_connection(
    port: str, baudrate: int = 1000000, timeout: float = 10.0
) -> serial.Serial:
    """
    Initialize serial connection to Arduino.

    Args:
        port: Serial port (e.g., '/dev/ttyACM0').
        baudrate: Baud rate for serial communication.
        timeout: Timeout in seconds for reading from serial.

    Returns:
        Initialized serial.Serial object.

    Raises:
        DataCollectionError: If connection fails.
    """
    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        logger.info(f"Connected to Arduino on {port} at {baudrate} baud.")
        return ser
    except serial.SerialException as e:
        logger.error(f"Failed to connect to Arduino on {port}: {e}")
        raise DataCollectionError(f"Failed to connect to Arduino on {port}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while connecting to Arduino: {e}")
        raise DataCollectionError(f"Unexpected error while connecting to Arduino: {e}")


def input_sample_metadata(timeout: float) -> Dict[str, Any]:
    """
    Prompt user for sample metadata: material, class, and optional comment.

    Args:
        timeout: Timeout in seconds for user input.

    Returns:
        Dictionary containing metadata.
    """
    while True:
        try:
            material_name = inputimeout(
                prompt="Material name (e.g., 'kitchen_tile', 'yoga_mat'): ",
                timeout=timeout,
            ).strip()
        except TimeoutOccurred:
            logger.error("Timeout occurred while waiting for input.")
            raise DataCollectionError("Timeout occurred while waiting for input.")
        if material_name and len(material_name) > 2:
            break
        logger.info("Wrong input. Please enter a meaningful material name (3+ chars)")

    while True:
        try:
            material_class = (
                inputimeout(prompt="Class (hard/medium/soft): ", timeout=timeout)
                .strip()
                .lower()
            )
        except TimeoutOccurred:
            logger.error("Timeout occurred while waiting for input.")
            raise DataCollectionError("Timeout occurred while waiting for input.")
        if material_class in ["hard", "medium", "soft"]:
            break
        logger.info("Wrong input. Please enter: hard, medium, or soft")

    try:
        comment = inputimeout(
            prompt="Comment (press Enter to skip): ", timeout=timeout
        ).strip()
    except TimeoutOccurred:
        logger.error("Timeout occurred while waiting for input.")
        raise DataCollectionError("Timeout occurred while waiting for input.")
    return {
        "material": material_name,
        "class": material_class,
        "comment": comment if comment else None,
    }


def get_arduino_data(ser: serial.Serial, timeout: float = 120.0) -> Dict[str, Any]:
    """
    Listen for Arduino data with non-blocking reads.

    This avoids the problem where large JSON payloads cause Arduino to block because
    PySerial's readline() is too slow to drain the serial buffer.

    Returns:
        Dictionary containing audio data and metadata from Arduino.

    Raises:
        DataCollectionError: If data is invalid, if a timeout occurs, or if serial read
        fails.
    """
    logger.info("Listening for trigger...")
    buffer = ""
    start_time = time.perf_counter()
    timeout_warning_logged = False

    while True:
        # Non-blocking read: read whatever is available
        if ser.in_waiting > 0:
            try:
                # Read up to 4KB at a time (small reads, keeps loop responsive)
                chunk = ser.read(min(ser.in_waiting, 4096))
                buffer += chunk.decode("utf-8", errors="ignore")

            except Exception as e:
                logger.error(f"Error reading from serial: {e}")
                raise DataCollectionError(f"Serial read error: {e}")

        # Process any complete JSON-like objects in buffer
        while True:
            # Look for complete JSON: starts with { and ends with }\n
            if buffer.startswith("{"):
                # Find closing brace followed by carriage return and newline
                # Arduino's serial.println() adds \r\n
                json_end = buffer.find("}\r\n")
                if json_end != -1:
                    json_str = buffer[: json_end + 1].strip()
                    buffer = buffer[json_end + 2 :]  # Remove processed data
                    logger.info("Received JSON data from Arduino.")
                    try:
                        data = json.loads(json_str)
                        logger.info("Parsed JSON data successfully.")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Invalid JSON from Arduino: {e}\nBuffer: {buffer[:200]}"
                        )
                        raise DataCollectionError(
                            f"Invalid JSON from Arduino: {e}\nBuffer: {buffer[:200]}"
                        )

                    _validate_audio_data(data)
                    return data
                else:
                    # JSON not complete yet, wait for more data
                    break
            else:
                # Buffer doesn't start with {, look for status messages
                line_end = buffer.find("\n")
                if line_end != -1:
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1 :]

                    # Log Arduino status messages
                    if any(
                        x in line
                        for x in [
                            "Drop object to trigger capture",
                            "TRIGGER DETECTED",
                            "RECORDING DATA",
                            "Max loop iteration time",
                            "READY FOR NEXT DROP",
                        ]
                    ):
                        logger.info(f"Arduino: {line}")
                    elif "END RECORDING" in line:
                        logger.error("Unexpected END RECORDING without valid JSON")
                        raise DataCollectionError(
                            "Unexpected END RECORDING without valid JSON"
                        )
                else:
                    # No complete line yet, wait for more data
                    break

        # Check timeout
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout / 2 and not timeout_warning_logged:
            logger.warning(
                f"No data has been received in {timeout / 2:.1f}s, still listening..."
            )
            timeout_warning_logged = True
        elif elapsed_time >= timeout:
            logger.error(f"Timeout: No data received in {elapsed_time:.1f}s")
            raise DataCollectionError("Timeout waiting for Arduino data")

        # Small 1ms sleep to prevent busy-waiting (CPU usage)
        time.sleep(0.001)


def _validate_audio_data(data: Dict[str, Any]) -> None:
    """
    Validate audio data from Arduino.

    Args:
        data: Raw data dictionary from Arduino.

    Raises:
        DataCollectionError: If validation fails.
    """
    # Check required fields
    required_fields = [
        "arduino_ide_version",
        "board",
        "microcontroller",
        "microphone",
        "sample_rate",
        "duration_seconds",
        "num_samples",
        "pre_trigger_samples",
        "post_trigger_samples",
        "trigger_threshold",
        "mic_pdm_gain",
        "overflow",
        "values",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        logger.error(f"Missing required fields in Arduino data: {missing}")
        raise DataCollectionError(f"Missing required fields: {missing}")

    # Validate sample count
    if len(data["values"]) != data["num_samples"]:
        logger.error(
            f"Sample count mismatch: expected {data['num_samples']}, "
            f"got {len(data['values'])}"
        )
        raise DataCollectionError(
            f"Sample count mismatch: expected {data['num_samples']}, "
            f"got {len(data['values'])}"
        )

    # Validate reasonable amplitude
    max_amplitude = max(abs(v) for v in data["values"])
    if max_amplitude < data["trigger_threshold"]:
        logger.error(
            f"Unexpected max amplitude ({max_amplitude}) < "
            f"trigger threshold ({data['trigger_threshold']}) in sample."
        )
        raise DataCollectionError(
            f"Unexpected max amplitude ({max_amplitude}) < "
            f"trigger threshold ({data['trigger_threshold']}) in sample."
        )

    # Raise if overflow detected
    if data.get("overflow"):
        logger.error("Audio buffer overflow detected. Discarding sample.")
        raise DataCollectionError("Audio buffer overflow detected. Discarding sample.")

    logger.info("Audio data validated successfully.")


def create_sample(
    metadata: Dict[str, Any], arduino_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert sample to dictionary format for JSON serialization.
    Follows the schema defined in data/raw/DATA_SCHEMA.md

    """
    sample_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "sample_id": sample_id,
            "timestamp": timestamp,
            "material": metadata["material"],
            "class": metadata["class"],
            "comment": metadata["comment"],
        },
        "hardware": {
            "board": arduino_data["board"],
            "microcontroller": arduino_data["microcontroller"],
            "microphone": arduino_data["microphone"],
            "arduino_ide_version": arduino_data["arduino_ide_version"],
            "mic_pdm_gain": arduino_data["mic_pdm_gain"],
        },
        "audio": {
            "sample_rate": arduino_data["sample_rate"],
            "num_samples": arduino_data["num_samples"],
            "duration_seconds": arduino_data["duration_seconds"],
            "pre_trigger_samples": arduino_data["pre_trigger_samples"],
            "post_trigger_samples": arduino_data["post_trigger_samples"],
            "trigger_threshold": arduino_data["trigger_threshold"],
            "overflow": arduino_data["overflow"],
            "values": arduino_data["values"],
        },
    }


def save_sample(sample: Dict[str, Any], output_dir: Path) -> None:
    """
    Save sample to JSON file.

    Args:
        sample: Dictionary with sample data.
        output_dir: Directory where the sample will be saved.

    Raises:
        DataCollectionError: If saving fails.
    """
    filename = f"{sample['metadata']['sample_id']}.json"
    filepath = output_dir / filename

    try:
        with open(filepath, "w") as f:
            json.dump(sample, f, indent=2)

        file_size_kb = filepath.stat().st_size / 1024
        logger.info(f"Saved: {filename} ({file_size_kb:.1f} KB)")

    except IOError as e:
        logger.error(f"Failed to save sample: {e}")
        raise DataCollectionError(f"Failed to save sample: {e}")


def wait_for_confirmation_with_buffer_drain(
    ser: serial.Serial, timeout: float = 120.0
) -> None:
    """
    Wait for user confirmation while continuously draining serial buffer.

    Prevents accidental Arduino triggers from blocking subsequent captures.
    Drains buffer every 500ms while waiting for user to press ENTER.
    """
    stop_drain = False

    def drain_buffer():
        """Continuously drain serial buffer until user is ready."""
        while not stop_drain:
            if ser.in_waiting > 0:
                drained = ser.read(min(ser.in_waiting, 4096))
                logger.debug(f"Drained {len(drained)} bytes during confirmation wait")
            time.sleep(0.5)  # Drain every 500ms

    # Start background drain thread (daemon so it exits with main thread)
    drain_thread = threading.Thread(target=drain_buffer, daemon=True)
    drain_thread.start()

    try:
        inputimeout(
            prompt=(
                "\nReady for drop? Press ENTER to start recording...\n"
                "(Wait for Arduino to be ready: Orange LED ON)\n"
            ),
            timeout=timeout,
        )
    except TimeoutOccurred:
        stop_drain = True
        raise DataCollectionError(
            "Timeout occurred while waiting for user confirmation."
        )
    finally:
        stop_drain = True
        drain_thread.join(timeout=1.0)
