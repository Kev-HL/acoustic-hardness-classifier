"""
Python module for data collection logic for the Accoustic Hardness Classifier Project.
"""

# Standard imports
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import serial
from inputimeout import inputimeout, TimeoutOccurred

# Set up logger
logger = logging.getLogger(__name__)


class DataCollectionError(Exception):
    """Custom exception for data collection errors."""

    pass


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
        raise DataCollectionError(f"Failed to connect to Arduino on {port}: {e}")


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
            raise DataCollectionError("Timeout occurred while waiting for input.")
        if material_class in ["hard", "medium", "soft"]:
            break
        logger.info("Wrong input. Please enter: hard, medium, or soft")

    try:
        comment = inputimeout(
            prompt="Comment (press Enter to skip): ", timeout=timeout
        ).strip()
    except TimeoutOccurred:
        raise DataCollectionError("Timeout occurred while waiting for input.")
    return {
        "material": material_name,
        "class": material_class,
        "comment": comment if comment else None,
    }


def get_arduino_data(ser: serial.Serial, timeout: float = 120.0) -> Dict[str, Any]:
    """
    Listen for Arduino trigger and capture audio data.

    Returns:
        Dictionary containing audio data and metadata from Arduino.

    Raises:
        DataCollectionError: If data is invalid.
    """
    logger.info("Listening for trigger...")
    buffer = ""
    start_time = time.perf_counter()
    serial_timeout = ser.timeout if ser.timeout is not None else 10.0

    while True:
        # Read line from Arduino
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            # Check for timeout warning
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > timeout / 2 and elapsed_time < (
                timeout / 2 + serial_timeout
            ):
                logger.warning(
                    f"No data received in more than {timeout/2:.2f} seconds."
                )
                logger.info("Continuing listening for trigger...")
            elif elapsed_time >= (timeout):
                raise DataCollectionError(
                    f"Timeout: No data received in {elapsed_time:.2f} seconds. "
                    "Check Arduino connection and settings."
                )
            continue

        # Print status messages
        if (
            "Drop object to trigger capture" in line
            or "TRIGGER DETECTED" in line
            or "RECORDING DATA" in line
            or "Max loop" in line
            or "READY FOR NEXT" in line
        ):
            logger.info(f"Arduino: {line}")
            continue

        # Capture JSON data
        if line.startswith("{") and line.endswith("}"):
            buffer = line
            logger.info("Received JSON data from Arduino.")
            break

        if "END RECORDING" in line:
            raise DataCollectionError(
                "There was an unexpected 'END RECORDING' or invalid JSON data. "
            )

    # Parse JSON
    try:
        data = json.loads(buffer)
        logger.info("Parsed JSON data successfully.")
    except json.JSONDecodeError as e:
        raise DataCollectionError(
            f"Invalid JSON from Arduino: {e}\nBuffer: {buffer[:200]}"
        )

    # Validate captured data
    _validate_audio_data(data)
    return data


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
        raise DataCollectionError(f"Missing required fields: {missing}")

    # Validate sample count
    if len(data["values"]) != data["num_samples"]:
        raise DataCollectionError(
            f"Sample count mismatch: expected {data['num_samples']}, "
            f"got {len(data['values'])}"
        )

    # Validate reasonable amplitude
    max_amplitude = max(abs(v) for v in data["values"])
    if max_amplitude < data["trigger_threshold"]:
        raise DataCollectionError(
            f"Unexpected max amplitude ({max_amplitude}) < "
            f"trigger threshold ({data['trigger_threshold']}) in sample."
        )

    # Raise if overflow detected
    if data.get("overflow"):
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
        raise DataCollectionError(f"Failed to save sample: {e}")
