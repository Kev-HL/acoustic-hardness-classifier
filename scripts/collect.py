"""
Data collection script for acoustic hardness classification.

Captures audio from Arduino Nano 33 BLE Sense Rev2 via serial connection,
validates, and saves samples as JSON with metadata.

Usage:
    python collect.py [output_dir DIR] [--port PORT] [--baud BAUDRATE]

Arguments:
    output_dir: Directory to save collected samples (default: data/raw)
    --port: Serial port for Arduino (default: /dev/ttyACM0) [Optional]
    --baud: Baud rate for serial communication (default: 1000000) [Optional]
"""

# Standard imports
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
from inputimeout import TimeoutOccurred, inputimeout

# Local imports
from ahc.collection import (
    DataCollectionError,
    InterruptDebouncer,
    create_sample,
    get_arduino_data,
    init_serial_connection,
    input_sample_metadata,
    save_sample,
    wait_for_confirmation_with_buffer_drain,
)

# Set up logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"data_collection_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# Constants
SAMPLE_TIMEOUT = 120  # seconds
SERIAL_TIMEOUT = 10.0  # seconds
DEFAULT_BAUDRATE = 1000000  # 1Mbps max rate for Nano 33 BLE Sense Rev2
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"  # Default serial port for Arduino

# Debouncer class to handle rapid interrupts in PyInstaller executables
interrupt_debouncer = InterruptDebouncer()


# Main function
def main() -> None:
    """Main data collection loop."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        prog="collect.py",
        description="Collect audio samples from Arduino Nano 33 BLE Sense Rev2",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve formatting
        epilog=f"""
            Default port if not specified: {DEFAULT_SERIAL_PORT}
            Default baudrate if not specified: {DEFAULT_BAUDRATE}
            Examples:
            python collect.py data/raw/phase1
            python collect.py data/raw/phase2 --port /dev/ttyACM0 --baud 1000000
        """,
    )
    parser.add_argument("output_dir", help="Output directory for samples")
    parser.add_argument(
        "--port",
        help="Serial port (can be checked in Arduino IDE)",
        default=DEFAULT_SERIAL_PORT,
    )
    parser.add_argument(
        "--baud",
        help="Baud rate for serial communication",
        type=int,
        default=DEFAULT_BAUDRATE,
    )
    args = parser.parse_args()

    # Print log starting message
    logger.info("=" * 60)
    logger.info("ACOUSTIC HARDNESS CLASSIFIER - DATA COLLECTION")
    logger.info("=" * 60)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    # Initialize serial object, samples counter, and sample start time
    ser = None
    samples_this_session = 0

    try:
        # Initialize serial connection
        ser = init_serial_connection(
            args.port, baudrate=int(args.baud), timeout=SERIAL_TIMEOUT
        )

        # Main collection loop
        while True:
            try:
                # Sample initial message
                logger.info("=" * 60)
                logger.info("NEW SAMPLE")
                logger.info("=" * 60 + "\n")

                # Get material metadata (name, class, comment)
                metadata = input_sample_metadata(SAMPLE_TIMEOUT)

                # Wait for user confirmation
                wait_for_confirmation_with_buffer_drain(ser, timeout=SAMPLE_TIMEOUT)

                # Capture audio from Arduino
                arduino_data = get_arduino_data(ser, SAMPLE_TIMEOUT)

                # Create and save sample after user confirmation
                sample = create_sample(metadata, arduino_data)
                logger.info("Sample created successfully")
                try:
                    discard_input = (
                        inputimeout(
                            prompt="\nValid sample? ENTER to keep, 'no' to discard: ",
                            timeout=SAMPLE_TIMEOUT,
                        )
                        .strip()
                        .lower()
                    )
                except TimeoutOccurred:
                    raise DataCollectionError(
                        "Timeout occurred while waiting for user input."
                    )
                if discard_input != "no":
                    save_sample(sample, output_dir)
                    samples_this_session += 1
                else:
                    logger.info("Sample discarded")

                # Ask if user wants to continue
                while True:
                    try:
                        continue_input = (
                            inputimeout(
                                prompt="\nContinue collecting? (y/n): ",
                                timeout=SAMPLE_TIMEOUT,
                            )
                            .strip()
                            .lower()
                        )
                    except TimeoutOccurred:
                        raise DataCollectionError(
                            "Timeout occurred while waiting for user input."
                        )
                    if continue_input in ["y", "yes", "n", "no"]:
                        break
                    logger.info("Wrong input. Please enter 'yes' or 'no'")

                if continue_input in ["n", "no"]:
                    break

            except DataCollectionError as e:
                if "timeout" in str(e).lower():
                    logger.error(f"\n\nERROR: {e}\n")
                    logger.error(
                        f"Timeout limit of {SAMPLE_TIMEOUT}s reached. "
                        "Exiting data collection."
                    )
                    break
                # If not exceeded timeout, log error and continue
                logger.error(f"\n\nERROR: {e}\n")
                logger.error("Please try again or adjust code or Arduino settings.")

        # Summary
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Samples collected this session: {samples_this_session}")
        samples_in_output_dir = len(list(output_dir.glob("*.json")))
        logger.info(f"Total samples in repository: {samples_in_output_dir}")
        logger.info(f"Output directory: {output_dir.resolve()}")
        logger.info("=" * 60)

    except DataCollectionError as e:
        logger.error(f"\n\nERROR: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n\nCollection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if ser and ser.is_open:
            ser.close()
            logger.info("Connection closed")


if __name__ == "__main__":
    main()
