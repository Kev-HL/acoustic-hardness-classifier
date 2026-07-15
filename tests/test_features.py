"""Unit tests for some of the features.py logic, located in src/ahc/features.py"""

# Third party imports
import numpy as np
import pytest

# Local imports
from ahc.features import (
    _compute_decay_ratio,
    _compute_features_single,
    _compute_settling_time,
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
        # Pre-trigger: 800 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high region then lower region
        pre_trigger = [5] * 800
        peak = [8000]
        first_half_decay = [200] * 1000
        second_half_decay = [50] * 999
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
        features = [
            "rms",
            "peak",
            "zcr",
            "settling_time",
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
        assert result["settling_time"] >= 0.0
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
        # Pre-trigger: 800 samples of low ambient noise
        # Impact: 1 large peak
        # Post-impact: high region then lower region
        pre_trigger = [5] * 800
        peak = [8000]
        first_half_decay = [200] * 1000
        second_half_decay = [50] * 999
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


class TestComputeSettlingTime:
    """Unit tests for the _compute_settling_time function."""

    def test_valid_input(self):
        """Test that _compute_settling_time returns a correct settling time."""
        sample_rate = 16000
        num_samples = 40000
        pre_trigger_samples = 800

        # Main impact amplitude
        impact_amp = 12000

        # Bounces after impact (time in seconds after main impact, amplitude)
        bounces = [
            (0.1, 5000),
            (0.15, 3000),
            (0.2, 1800),
        ]

        # Ring decay rate (how quickly the ringing decays)
        ring_decay = 0.02

        # Signal array initialization
        signal = np.zeros(num_samples, dtype=np.float32)

        # Small background noise
        rng = np.random.default_rng(1234)
        signal += rng.normal(0, 2, num_samples)

        # Main impact
        impact_idx = pre_trigger_samples
        signal[impact_idx] += impact_amp

        # Main ringing
        t = np.arange(num_samples - impact_idx) / sample_rate
        ring = impact_amp * 0.35 * np.exp(-t / ring_decay) * np.sin(2 * np.pi * 500 * t)
        signal[impact_idx:] += ring

        # Add bounces with ringing
        for time_s, amp in bounces:
            idx = impact_idx + int(time_s * sample_rate)
            signal[idx] += amp

            tt = np.arange(num_samples - idx) / sample_rate
            signal[idx:] += (
                amp
                * 0.35
                * np.exp(-tt / ring_decay * 0.9)
                * np.sin(2 * np.pi * 500 * tt)
            )

        # Convert signal to int16
        signal = signal.astype(np.int16)

        # Compute settling time using the function
        result = _compute_settling_time(signal, sample_rate, pre_trigger_samples)

        # Get approximate result based on the last bounce time and sample rate
        approx_settling_time = impact_idx / sample_rate + bounces[-1][0]

        assert isinstance(result, float)
        assert result >= 0.0
        assert result == pytest.approx(approx_settling_time, rel=1e-1)

    def test_never_settling(self):
        """
        Test that _compute_settling_time returns end of signal with a never settling
        signal.

        Mimicked with many bounces and high ringing.
        """
        sample_rate = 16000
        num_samples = 40000
        pre_trigger_samples = 800

        # Main impact amplitude
        impact_amp = 12000

        # Bounces after impact (time in seconds after main impact, amplitude)
        bounces = [
            (0.5, 10000),
            (0.95, 8000),
            (1.35, 6000),
            (1.7, 4000),
            (2.0, 3000),
            (2.25, 2000),
        ]

        # Ring decay rate (how quickly the ringing decays)
        ring_decay = 0.2  # VERY HIGH RINGING, WILL NOT SETTLE

        # Signal array initialization
        signal = np.zeros(num_samples, dtype=np.float32)

        # Small background noise
        rng = np.random.default_rng(1234)
        signal += rng.normal(0, 2, num_samples)

        # Main impact
        impact_idx = pre_trigger_samples
        signal[impact_idx] += impact_amp

        # Main ringing
        t = np.arange(num_samples - impact_idx) / sample_rate
        ring = impact_amp * 0.35 * np.exp(-t / ring_decay) * np.sin(2 * np.pi * 500 * t)
        signal[impact_idx:] += ring

        # Add bounces with ringing
        for time_s, amp in bounces:
            idx = impact_idx + int(time_s * sample_rate)
            signal[idx] += amp

            tt = np.arange(num_samples - idx) / sample_rate
            signal[idx:] += (
                amp
                * 0.35
                * np.exp(-tt / ring_decay * 0.9)
                * np.sin(2 * np.pi * 500 * tt)
            )

        # Convert signal to int16
        signal = signal.astype(np.int16)

        # Compute settling time using the function
        result = _compute_settling_time(signal, sample_rate, pre_trigger_samples)

        # Get approximate result based on the end of the signal
        approx_settling_time = (len(signal) - pre_trigger_samples) / sample_rate

        assert isinstance(result, float)
        assert result >= 0.0
        assert result == pytest.approx(approx_settling_time, rel=1e-1)

    def test_noise_signal(self):
        """
        Test that _compute_settling_time returns end of signal with a pure noise signal.

        Should default to end of signal if the signal is very noisy and never settles.
        """
        sample_rate = 16000
        num_samples = 40000
        pre_trigger_samples = 800

        # Signal array initialization
        signal = np.zeros(num_samples, dtype=np.float32)

        # Small background noise
        rng = np.random.default_rng(1234)
        signal += rng.normal(0, 2, num_samples)

        # Convert signal to int16
        signal = signal.astype(np.int16)

        # Compute settling time using the function
        result = _compute_settling_time(signal, sample_rate, pre_trigger_samples)

        # Get approximate result based end of signal
        approx_settling_time = (len(signal) - pre_trigger_samples) / sample_rate

        assert isinstance(result, float)
        assert result >= 0.0
        assert result == pytest.approx(approx_settling_time, rel=1e-1)

    def test_zero_signal(self):
        """
        Test that _compute_settling_time returns end of signal with all zeros.

        Should default to end of signal.
        """
        sample_rate = 16000
        num_samples = 40000
        pre_trigger_samples = 800

        # Signal array initialization
        signal = np.zeros(num_samples, dtype=np.int16)

        # Compute settling time using the function
        result = _compute_settling_time(signal, sample_rate, pre_trigger_samples)

        # Get approximate result based end of signal
        approx_settling_time = (len(signal) - pre_trigger_samples) / sample_rate

        assert isinstance(result, float)
        assert result >= 0.0
        assert result == pytest.approx(approx_settling_time, rel=1e-1)

    def test_short_signal(self, caplog):
        """Test that _compute_settling_time warns for short signals."""
        pre_trigger = [5] * 20
        peak = [6000]
        decay = [200] * 10
        signal = np.array(pre_trigger + peak + decay)
        pre_trigger_samples = len(pre_trigger)
        sample_rate = 16000

        with caplog.at_level("WARNING"):
            result = _compute_settling_time(signal, sample_rate, pre_trigger_samples)

        assert isinstance(result, float)
        assert result >= 0.0
        assert "Signal length is shorter than expected" in caplog.text

    def test_invalid_sample_rate(self):
        """Test that _compute_settling_time raises with sample_rate <= 0."""
        pre_trigger = [5] * 200
        first_peak = [6000]
        first_decay = [200] * 50
        signal = np.array(pre_trigger + first_peak + first_decay)
        pre_trigger_samples = len(pre_trigger)
        sample_rate = -1  # Invalid sample rate

        with pytest.raises(ValueError, match="Sample rate must be a positive integer."):
            _compute_settling_time(signal, sample_rate, pre_trigger_samples)

    def test_invalid_pre_trigger_samples(self):
        """Test that _compute_settling_time raises with invalid pre_trigger_samples."""
        pre_trigger = [5] * 200
        first_peak = [6000]
        first_decay = [200] * 50
        signal = np.array(pre_trigger + first_peak + first_decay)
        pre_trigger_samples = len(signal) + 1  # Invalid
        sample_rate = 16000

        with pytest.raises(
            ValueError,
            match=(
                "Pre-trigger samples must be a positive integer "
                "less than the length of the signal."
            ),
        ):
            _compute_settling_time(signal, sample_rate, pre_trigger_samples)

    def test_negative_pre_trigger_samples(self):
        """Test that _compute_settling_time raises with negative pre_trigger_samples."""
        pre_trigger = [5] * 200
        first_peak = [6000]
        first_decay = [200] * 50
        signal = np.array(pre_trigger + first_peak + first_decay)
        pre_trigger_samples = -1  # Invalid
        sample_rate = 16000

        with pytest.raises(
            ValueError,
            match=(
                "Pre-trigger samples must be a positive integer "
                "less than the length of the signal."
            ),
        ):
            _compute_settling_time(signal, sample_rate, pre_trigger_samples)
