"""Unit tests for some of the features.py logic, located in src/ahc/features.py"""

# Third party imports
import numpy as np
import pytest

# Local imports
from ahc.features import (
    _compute_decay_ratio,
    _compute_features_single,
    compute_features,
)


class TestComputeDecayRatio:
    """Unit tests for the _compute_decay_ratio function."""

    def test_valid_input(self):
        """Test that _compute_decay_ratio returns the correct decay ratio."""
        # Pre-trigger: 8 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high region then lower region, so mav2 < mav1
        pre_trigger = [5] * 8
        peak = [8000]  # threshold would be 1/8 of 8000 = 1000
        first_half_decay = [200] * 100
        second_half_decay = [50] * 100
        signal = np.array(pre_trigger + peak + first_half_decay + second_half_decay)
        pre_trigger_samples = len(pre_trigger)
        expected_decay_ratio = 50 / 200  # mav2/mav1 = 0.25

        result = _compute_decay_ratio(signal, pre_trigger_samples)

        assert result == pytest.approx(expected_decay_ratio)

    def test_no_decay_signal(self):
        """Test that _compute_decay_ratio returns 1.0 for a no decaying signal."""
        # Pre-trigger: 8 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high enough that does not go over the threshold
        pre_trigger = [5] * 8
        peak = [1000]  # threshold would be 1/8 of 1000 = 125
        first_half_decay = [200] * 100
        second_half_decay = [50] * 100
        signal = np.array(pre_trigger + peak + first_half_decay + second_half_decay)
        pre_trigger_samples = len(pre_trigger)
        expected_decay_ratio = 1.0  # w_length < 100, so return 1.0

        result = _compute_decay_ratio(signal, pre_trigger_samples)

        assert result == pytest.approx(expected_decay_ratio)

    def test_silent_signal(self):
        """Test that _compute_decay_ratio returns 0.0 for a silent signal."""
        # Test for logical safeguard, case cannot happen with our data collection setup.
        # Pre-trigger: 8 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high enough that does not go over the threshold
        pre_trigger = [0] * 8
        peak = [8]  # threshold would be 1/8 of 8 = 1
        first_half_decay = [0] * 100
        second_half_decay = [0] * 100
        signal = np.array(pre_trigger + peak + first_half_decay + second_half_decay)
        pre_trigger_samples = len(pre_trigger)
        expected_decay_ratio = 0.0  # silent signal should return 0.0

        result = _compute_decay_ratio(signal, pre_trigger_samples)

        assert result == pytest.approx(expected_decay_ratio)


class TestComputeFeaturesSingle:
    """Unit tests for the _compute_features_single function."""

    def test_valid_input(self):
        """Test that _compute_features_single returns the correct features."""
        # Pre-trigger: 8 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high region then lower region
        pre_trigger = [5] * 8
        peak = [8000]  # threshold would be 1/8 of 8000 = 1000
        first_half_decay = [200] * 100
        second_half_decay = [50] * 100
        signal = np.array(pre_trigger + peak + first_half_decay + second_half_decay)
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                "sample_rate": 16000,
                "num_samples": 40000,
                "duration_seconds": 2.5,
                "pre_trigger_samples": 800,
                "values": signal,
            },
        }
        features = [
            "rms",
            "peak",
            "zcr",
            "decay_ratio",
            "crest_factor",
            "spectral_flatness",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff_95",
        ]
        signal_without_dc = signal - np.mean(signal)

        result = _compute_features_single(sample)

        assert isinstance(result, dict)
        assert len(result) == len(features)
        assert result.keys() == set(features)
        assert result["decay_ratio"] >= 0.0
        assert result["peak"] == np.max(np.abs(signal_without_dc))
        assert result["zcr"] >= 0.0 and result["zcr"] <= 1.0

    def test_missing_audio(self):
        """Test that _compute_features_single raises ValueError for missing audio."""
        sample = {
            "metadata": {
                "sample_id": "1",
            },
        }

        with pytest.raises(
            ValueError, match="Sample 1 has invalid format and/or content."
        ):
            _compute_features_single(sample)

    def test_missing_audio_values(self):
        """
        Test that _compute_features_single raises ValueError for missing audio values.
        """
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                "sample_rate": 16000,
                "num_samples": 40000,
                "duration_seconds": 2.5,
                "pre_trigger_samples": 800,
                # Missing 'values' key
            },
        }

        with pytest.raises(
            ValueError, match="Sample 1 has invalid format and/or content."
        ):
            _compute_features_single(sample)

    def test_empty_audio_values(self):
        """
        Test that _compute_features_single raises ValueError for empty audio values.
        """
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                "sample_rate": 16000,
                "num_samples": 40000,
                "duration_seconds": 2.5,
                "pre_trigger_samples": 800,
                "values": [],  # Empty values
            },
        }

        with pytest.raises(
            ValueError, match="Sample 1 has invalid format and/or content."
        ):
            _compute_features_single(sample)

    def test_missing_sample_rate(self):
        """
        Test that _compute_features_single raises ValueError for missing sample rate.
        """
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                # Missing 'sample_rate' key
                "num_samples": 40000,
                "duration_seconds": 2.5,
                "pre_trigger_samples": 800,
                "values": [0] * 40000,
            },
        }

        with pytest.raises(
            ValueError, match="Sample 1 has invalid format and/or content."
        ):
            _compute_features_single(sample)

    def test_missing_pre_trigger_samples(self):
        """
        Test that _compute_features_single raises ValueError for missing
        pre_trigger_samples.
        """
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                "sample_rate": 16000,
                "num_samples": 40000,
                "duration_seconds": 2.5,
                # Missing 'pre_trigger_samples' key
                "values": [0] * 40000,
            },
        }

        with pytest.raises(
            ValueError, match="Sample 1 has invalid format and/or content."
        ):
            _compute_features_single(sample)


class TestComputeFeatures:
    """Unit tests for the compute_features function."""

    def test_valid_input(self):
        """Test that compute_features returns a list of feature dictionaries."""
        # Pre-trigger: 8 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high region then lower region
        pre_trigger = [5] * 8
        peak = [8000]  # threshold would be 1/8 of 8000 = 1000
        first_half_decay = [200] * 100
        second_half_decay = [50] * 100
        signal = np.array(pre_trigger + peak + first_half_decay + second_half_decay)
        sample = {
            "metadata": {
                "sample_id": "1",
            },
            "audio": {
                "sample_rate": 16000,
                "num_samples": len(signal),
                "duration_seconds": len(signal) / 16000,
                "pre_trigger_samples": len(pre_trigger),
                "values": signal,
            },
        }

        result = compute_features([sample])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "features" in result[0]

    def test_empty_input(self):
        """Test that compute_features raises ValueError for empty input."""
        with pytest.raises(ValueError):
            compute_features([])
