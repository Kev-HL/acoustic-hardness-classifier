"""
Python module for feature extraction logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from ahc.signal_processing import remove_dc_offset

# Set up logger
logger = logging.getLogger(__name__)


def _compute_features_single(sample: dict) -> dict:
    """
    Compute basic acoustic features of a single recording sample.

    Args:
        sample (dict): A dictionary containing the audio values, where the values are
        stored under ["audio"]["values"] and the sample rate in Hz is stored under
        ["audio"]["sample_rate"].

    Returns:
        dict: A dictionary containing computed features

    Raises:
        ValueError: If the sample is missing required keys or has invalid content.

    Usage:
        sample_features = _compute_features_single(sample)
        or indirectly:
        samples = compute_features(samples)
    """
    # Safeguard against missing or malformed data
    if (
        "audio" not in sample
        or "values" not in sample["audio"]
        or not isinstance(sample["audio"]["values"], (list, np.ndarray))
        or len(sample["audio"]["values"]) == 0
        or "sample_rate" not in sample["audio"]
        or "pre_trigger_samples" not in sample["audio"]
    ):
        sample_id = sample.get("metadata", {}).get("sample_id", "unknown")
        logger.error(f"Sample {sample_id} has invalid format and/or content.")
        raise ValueError(f"Sample {sample_id} has invalid format and/or content.")

    # Epsilon small value to avoid division by zero in calculations
    eps = 1e-12

    # Extract audio values and sample rate
    values = np.array(sample["audio"]["values"])
    sample_rate = sample["audio"]["sample_rate"]
    pre_trigger_samples = sample["audio"]["pre_trigger_samples"]

    # Remove DC offset (mean) to center the signal around zero
    values = remove_dc_offset(values)

    # TIME-DOMAIN FEATURES
    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(values**2))
    # Peak Absolute Amplitude
    peak = np.max(np.abs(values))
    # Zero Crossing Rate (ZCR) (normalized)
    zcr = np.mean(np.abs(np.diff(np.signbit(values).astype(int))))
    # Decay Ratio (post-impact)
    decay_ratio = _compute_decay_ratio(values, pre_trigger_samples)
    # Crest Factor (peak-to-RMS ratio)
    crest_factor = peak / (rms + eps)

    # FREQUENCY DOMAIN FEATURES (Fast Fourier Transform magnitude)
    # Only half (positive freqs) considered, other half is omitted (negative mirror)
    fft_mag_pos = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values), 1 / sample_rate)
    # Spectral flatness or wiener entropy:
    # Geometric mean / arithmetic mean of the spectrum
    power_spectrum = fft_mag_pos**2
    log_mean = np.mean(np.log(power_spectrum[1:] + eps))
    arithmetic_mean = np.mean(power_spectrum[1:])
    spectral_flatness = np.exp(log_mean) / (arithmetic_mean + eps)

    # SPECTRAL FEATURES
    eps = 1e-12
    total_magnitude = np.sum(fft_mag_pos)
    # Spectral Centroid, center of "mass" of the spectrum
    spectral_centroid = np.sum(freqs * fft_mag_pos) / (total_magnitude + eps)
    # Spectral Bandwidth, spread of the spectrum around the centroid
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_mag_pos)
        / (total_magnitude + eps)
    )
    # Spectral Rolloff (95%), frequency below which 95% of the magnitude is contained
    rolloff_95_idx = np.where(np.cumsum(fft_mag_pos) >= 0.95 * total_magnitude)[0]
    if len(rolloff_95_idx) == 0:
        spectral_rolloff_95 = 0.0
    else:
        spectral_rolloff_95 = freqs[rolloff_95_idx[0]]

    # Return features as a dictionary
    return {
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "decay_ratio": decay_ratio,
        "crest_factor": crest_factor,
        "spectral_flatness": spectral_flatness,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff_95": spectral_rolloff_95,
    }


def _compute_decay_ratio(signal: np.ndarray, pre_trigger_samples: int) -> float:
    """
    Compute decay ratio as MAV(second half) / MAV(first half) of the post-impact signal.
    Returns a value in [0, 1] where values close to 0 indicate fast decay (soft surface)
    and values close to 1 indicate slow decay (hard surface).
    Note: Using MAV instead of RMS to reduce compute cost (to make it comparable to the
    feature if later on computed on Arduino).
    """
    # Compute absolute value of the signal
    abs_signal = np.abs(signal)

    # Extract noise samples using the first 90% of pre-trigger samples, to avoid
    # including the impact in the noise estimate as well as any precursor waves
    amb_noise_samples = int(0.9 * pre_trigger_samples)

    # Estimate noise floor from ambient_noise_samples
    noise_floor = np.mean(abs_signal[:amb_noise_samples])

    # Find peak value and its index
    peak_index = np.argmax(abs_signal)
    max_peak = abs_signal[peak_index]

    # Set threshold to be the greater of 1/8 of the peak or 3 times the noise floor
    threshold = max(max_peak / 8.0, noise_floor * 3)

    # Find where signal drops below threshold after peak
    total_samples = len(abs_signal)
    dynamic_start = total_samples - 1  # fallback: end of signal
    for i in range(peak_index, total_samples - 10):
        # Check window of 10 samples to avoid transient dips below threshold
        if np.max(abs_signal[i : i + 10]) < threshold:
            dynamic_start = i
            break

    # Need enough samples remaining for two meaningful windows
    remaining = total_samples - dynamic_start
    w_length = remaining // 2
    # If almost no remaining signal, assume no decay (1.0)
    if w_length < 100:
        return 1.0

    # Compute windows of same size and their mean absolute values (MAV)
    w1 = abs_signal[dynamic_start : dynamic_start + w_length]
    w2 = abs_signal[dynamic_start + w_length : dynamic_start + 2 * w_length]

    mav1 = np.mean(w1)
    mav2 = np.mean(w2)

    # Logic safeguard, if 1st window is silent, assume soft target (return 0.0)
    # Note: cannot happen because of our data collection setup
    if mav1 < 1e-12:
        return 0.0

    return float(mav2 / mav1)


def compute_features(samples: list[dict]) -> list[dict]:
    """
    Compute features for a list of audio samples. Assumes that all samples have the same
    sample rate and duration.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples,
        where each sample is a dictionary containing the audio values, where the values
        are stored under ["audio"]["values"] and the sample rate in Hz is stored under
        ["audio"]["sample_rate"].

    Returns:
        list of dict: The same list of dictionaries, but with an additional key
        "features" in each sample dictionary, containing the computed features.

    Usage (mutates the input list in place):
        samples_with_features = compute_features(samples)
    """
    # Safeguard against empty input
    if not samples:
        logger.error("Input samples list is empty.")
        raise ValueError("Input samples list is empty.")

    # Compute features for each sample and add them to the sample dictionary
    for sample in samples:
        sample["features"] = _compute_features_single(sample)

    return samples
