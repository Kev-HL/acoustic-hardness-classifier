"""
Python module for signal processing logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)


def remove_dc_offset(signal: np.ndarray | list) -> np.ndarray:
    """
    Remove DC offset from a signal by subtracting the mean.

    Args:
        signal (np.ndarray or list): The input audio signal.

    Returns:
        np.ndarray: The DC offset removed signal.

    Raises:
        ValueError: If the input signal is empty.
    """
    if len(signal) == 0:
        logger.error("Input signal is empty.")
        raise ValueError("Input signal is empty.")

    return np.array(signal) - np.mean(signal)
