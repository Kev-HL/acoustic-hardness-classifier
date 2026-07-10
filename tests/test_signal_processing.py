"""
Unit tests for some of the signal_processing.py logic,
located in src/ahc/signal_processing.py
"""

# Standard imports
import random

# Third party imports
import numpy as np
import pytest

# Local imports
from ahc.signal_processing import remove_dc_offset


class TestRemoveDCOffset:
    """Test cases for the remove_dc_offset function."""

    def test_remove_dc_offset(self):
        """Test remove_dc_offset with a sample signal."""
        signal = [random.uniform(-50.0, 100.0) for _ in range(5)]
        result = remove_dc_offset(signal)
        assert np.mean(result) == pytest.approx(0.0, abs=1e-6)

    def test_remove_dc_offset_empty_signal(self):
        """Test remove_dc_offset with an empty signal."""
        with pytest.raises(ValueError, match="Input signal is empty."):
            remove_dc_offset([])
