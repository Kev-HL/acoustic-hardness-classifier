"""
Python module for data analysis logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import logging

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# Local imports
from ahc.signal_processing import remove_dc_offset

# Set up logger
logger = logging.getLogger(__name__)


def compute_class_statistics(samples: list[dict]) -> dict:
    """
    Compute summary statistics for features by class.

    IMPORTANT: This function assumes that features have already been computed for each
    sample, and that all samples have the same set of features, and the same sample rate
    and duration. Also assumes that first sample has the correct set of features.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples,
        where each sample is a dictionary with metadata, audio data, and computed
        features.

    Returns:
        dict: A dictionary containing summary statistics for each class.

    Raises:
        ValueError: If no computed features are found in the samples, or if any sample
        is missing required metadata or features.

    Usage:
        class_stats = compute_class_statistics(samples)
    """
    # Safeguard against empty input
    if not samples or len(samples) == 0:
        logger.error("Input samples list is empty.")
        raise ValueError("Input samples list is empty.")

    # Safeguard against features not available
    features = samples[0]["features"].keys() if samples else []
    if not features:
        logger.error("No computed features found in samples.")
        raise ValueError("No computed features found in samples.")

    # Initialize dict to store statistics
    class_values = {}
    class_stats = {}

    for sample in samples:
        if (
            "metadata" not in sample
            or "class" not in sample["metadata"]
            or "features" not in sample
        ):
            sample_id = sample.get("metadata", {}).get("sample_id", "unknown")
            logger.error(f"Sample {sample_id} missing required metadata or features.")
            raise ValueError(
                f"Sample {sample_id} missing required metadata or features."
            )
        if sample["metadata"]["class"] not in class_values:
            class_values[sample["metadata"]["class"]] = {
                f"{feature}": [] for feature in features
            }
        for feature in features:
            if feature not in sample["features"]:
                sample_id = sample.get("metadata", {}).get("sample_id", "unknown")
                logger.error(
                    f"Sample {sample_id} missing computed feature '{feature}'."
                )
                raise ValueError(
                    f"Sample {sample_id} missing computed feature '{feature}'."
                )

            class_values[sample["metadata"]["class"]][f"{feature}"].append(
                sample["features"][feature]
            )

    for class_name in sorted(class_values.keys()):
        class_stats[class_name] = {}
        for feature in features:
            values = class_values[class_name][f"{feature}"]
            class_stats[class_name][f"{feature}"] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                "min": np.min(values),
                "max": np.max(values),
            }

    return class_stats


def _validate_release_method_samples(
    samples_A: list[dict], samples_B: list[dict]
) -> tuple[set, set, set]:
    """
    Validate that the samples from two release methods have matching:
    - classes
    - features
    - materials

    Args:
        samples_A: Samples from method A.
        samples_B: Samples from method B.

    Raises:
        ValueError: If class names, features, or materials do not match between sets.
        KeyError: If any sample is missing required metadata or features.
    """
    classes_A = set()
    classes_B = set()
    features_A = set()
    features_B = set()
    materials_per_class_A = set()
    materials_per_class_B = set()

    if not samples_A or not samples_B:
        logger.error("One or both sample sets are empty.")
        raise ValueError("One or both sample sets are empty.")

    if len(samples_A) != len(samples_B):
        logger.warning(
            "Sample sets have different lengths: "
            f"samples_A={len(samples_A)}, samples_B={len(samples_B)}"
        )

    for sample in samples_A:
        classes_A.add(sample["metadata"]["class"])
        features_A.update(sample["features"].keys())
        materials_per_class_A.add(
            (sample["metadata"]["material"], sample["metadata"]["class"])
        )
    for sample in samples_B:
        classes_B.add(sample["metadata"]["class"])
        features_B.update(sample["features"].keys())
        materials_per_class_B.add(
            (sample["metadata"]["material"], sample["metadata"]["class"])
        )

    # Safeguard against class mismatch between methods
    if classes_A != classes_B:
        logger.error("Class names in samples_A and samples_B do not match.")
        raise ValueError("Class names in samples_A and samples_B do not match.")

    # Safeguard against features mismatch between methods
    if features_A != features_B:
        logger.error("Features in samples_A and samples_B do not match.")
        raise ValueError("Features in samples_A and samples_B do not match.")

    # Safeguard against materials mismatch between methods
    if materials_per_class_A != materials_per_class_B:
        logger.error("Materials in samples_A and samples_B do not match.")
        raise ValueError("Materials in samples_A and samples_B do not match.")

    return classes_A, features_A, materials_per_class_A


def compare_release_methods_bias(
    samples_A: list[dict],
    samples_B: list[dict],
    method_A_name: str = "Device",
    method_B_name: str = "Manual",
) -> pd.DataFrame:
    """
    Function to quantitatively compare two release methods using normalized delta.

    For each class and feature, computes:
      - mean and std for each method
      - mean delta (A - B)
      - pooled std: sqrt((std_A^2 + std_B^2) / 2)
      - normalized delta: |mean_delta| / pooled std

    A normalized_delta << 1 means the method difference is smaller than
    within-class variation — safe to use either method.
    A normalized_delta >> 1 means the method introduces a consistent shift
    that exceeds natural variation — investigate further.

    Args:
        samples_A: Samples from method A.
        samples_B: Samples from method B.
        method_A_name: Label for method A.
        method_B_name: Label for method B.

    Returns:
        pd.DataFrame with columns:
            class, feature, mean_A, std_A, mean_B, std_B,
            mean_delta, normalized_delta
    """
    # Define a small epsilon to avoid division by zero in calculations
    eps = 1e-12

    # Validate and extract classes and features
    class_names, features, _ = _validate_release_method_samples(samples_A, samples_B)

    results = []
    for class_name in class_names:
        samples_A_class = [s for s in samples_A if s["metadata"]["class"] == class_name]
        samples_B_class = [s for s in samples_B if s["metadata"]["class"] == class_name]

        for feature in features:
            vals_A = np.array([s["features"][feature] for s in samples_A_class])
            vals_B = np.array([s["features"][feature] for s in samples_B_class])

            mean_A, std_A = np.mean(vals_A), np.std(vals_A, ddof=1)
            mean_B, std_B = np.mean(vals_B), np.std(vals_B, ddof=1)
            mean_delta = mean_A - mean_B

            # Normalized delta: method shift relative to within-class spread.
            # Uses the pooled std as the reference scale.
            # If both stds are ~0 (perfect reproducibility), avoid division by zero.
            denom = np.sqrt((std_A**2 + std_B**2) / 2)
            normalized_delta = abs(mean_delta) / denom if denom > eps else 0.0

            results.append(
                {
                    "class": class_name,
                    "feature": feature,
                    f"mean_{method_A_name}": mean_A,
                    f"std_{method_A_name}": std_A,
                    f"mean_{method_B_name}": mean_B,
                    f"std_{method_B_name}": std_B,
                    "mean_delta": mean_delta,
                    "normalized_delta": normalized_delta,
                }
            )

    return pd.DataFrame(results)


def compare_release_methods_reproducibility(
    samples_A: list[dict],
    samples_B: list[dict],
) -> pd.DataFrame:
    """
    Compare reproducibility in two release methods.

    For each class, feature, and method, computes the coefficient of variation (CV) for
    each material, then averages the CVs across materials.
    Finally, computes the relative CV between the two methods (CV_A / CV_B).

    If relative CV == 1, method A is as reproducible as method B.
    If relative CV < 1, method A is more reproducible than method B.
    If relative CV > 1, method A is less reproducible than method B.

    Args:
        samples_A: Samples from method A.
        samples_B: Samples from method B.

    Returns:
        pd.DataFrame with columns:
            feature, class, relative_cv
    """
    # Define a small epsilon to avoid division by zero in calculations
    eps = 1e-12

    # Validate and extract classes, features and materials
    class_names, features, material_names = _validate_release_method_samples(
        samples_A, samples_B
    )

    results = []
    for class_name in class_names:
        samples_A_class = [s for s in samples_A if s["metadata"]["class"] == class_name]
        samples_B_class = [s for s in samples_B if s["metadata"]["class"] == class_name]
        materials_class = set()
        for m in material_names:
            if m[1] == class_name:
                materials_class.add(m[0])

        for feature in features:
            cv_per_mat_A = []
            cv_per_mat_B = []

            for material in materials_class:
                vals_A = np.array(
                    [
                        s["features"][feature]
                        for s in samples_A_class
                        if s["metadata"]["material"] == material
                    ]
                )
                vals_B = np.array(
                    [
                        s["features"][feature]
                        for s in samples_B_class
                        if s["metadata"]["material"] == material
                    ]
                )
                # Compute coefficient of variation (CV) for each method and material
                # Use absolute mean to avoid issues with features that can be negative
                abs_mean_A = abs(np.mean(vals_A))
                abs_mean_B = abs(np.mean(vals_B))
                cv_A = np.std(vals_A, ddof=1) / abs_mean_A if abs_mean_A > eps else 0.0
                cv_B = np.std(vals_B, ddof=1) / abs_mean_B if abs_mean_B > eps else 0.0
                cv_per_mat_A.append(cv_A)
                cv_per_mat_B.append(cv_B)
            avg_cv_A = np.mean(cv_per_mat_A)
            avg_cv_B = np.mean(cv_per_mat_B)
            relative_cv = avg_cv_A / avg_cv_B if avg_cv_B > eps else float("inf")

            results.append(
                {"feature": feature, "class": class_name, "relative_cv": relative_cv}
            )

    return pd.DataFrame(results)


def compute_ambient_noise_level(samples: list[dict], duration: float) -> dict:
    """
    Compute the ambient noise level for each class based on a set duration.

    Duration is passed as an argument instead of using the full pre-trigger window to
    allow for flexibility when we want to avoid catching any precursor wave or main
    impact values in the noise estimate.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples.
        duration (float): Duration in seconds to use for ambient noise estimation.

    Returns:
        dict: A dictionary containing the mean and standard deviation of the ambient
        noise level across all samples.
    """
    # Safeguard against empty input
    if not samples:
        logger.error("Input samples list is empty.")
        raise ValueError("Input samples list is empty.")

    # Safeguard against non-positive duration
    if duration <= 0:
        logger.error("Duration must be a positive value.")
        raise ValueError("Duration must be a positive value.")

    amb_noise_mean_values = []
    amb_noise_std_values = []
    max_amb_noise_value = -np.inf
    min_amb_noise_value = +np.inf
    for sample in samples:
        signal = remove_dc_offset(sample["audio"]["values"])
        amb_noise_samples = int(duration * sample["audio"]["sample_rate"])
        amb_noise_signal = signal[:amb_noise_samples]
        amb_noise_mean = np.mean(amb_noise_signal)
        amb_noise_std = np.std(amb_noise_signal, ddof=1)
        amb_noise_mean_values.append(amb_noise_mean)
        amb_noise_std_values.append(amb_noise_std)
        if np.max(amb_noise_signal) > max_amb_noise_value:
            max_amb_noise_value = np.max(amb_noise_signal)
        if np.min(amb_noise_signal) < min_amb_noise_value:
            min_amb_noise_value = np.min(amb_noise_signal)
    avg_amb_noise_mean = np.mean(amb_noise_mean_values)
    avg_amb_noise_std = np.sqrt(np.mean(np.array(amb_noise_std_values) ** 2))

    return {
        "mean": avg_amb_noise_mean,
        "std": avg_amb_noise_std,
        "range": (min_amb_noise_value, max_amb_noise_value),
    }


def plot_time_domain(samples, plot_window=None, legend=False, classes_axes_map=None):
    """
    Plot raw waveforms for each class.

    Args:
        samples: List of sample dicts
        plot_window: Plot window in seconds. If None, plot full duration.
        legend: True/False Show sample IDs in legend
        classes_axes_map: Dict mapping class names to subplot indices

    Note: Assumes samples have been validated, and all samples have the same sample rate
    and duration.
    """
    # Default mapping of classes to subplot indices if not provided
    if classes_axes_map is None:
        classes_axes_map = {"hard": 0, "medium": 1, "soft": 2}

    # Get the duration to plot, capped at the sample duration boundaries
    if plot_window is None:
        plot_window = [0.0, 0.0]  # Defaults to full duration
    window_start, window_end = plot_window
    signal_duration = samples[0]["audio"]["duration_seconds"]
    window_start = max(0.0, window_start)
    window_end = min(window_end, signal_duration)
    if window_end <= window_start:
        window_start = 0.0
        window_end = signal_duration

    # Get the sample rate and number of samples to plot
    sample_rate = samples[0]["audio"]["sample_rate"]
    num_samples_start = int(window_start * sample_rate)
    num_samples_end = int(window_end * sample_rate)
    num_samples = num_samples_end - num_samples_start

    time = np.linspace(window_start, window_end, num_samples)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Time-Domain Audio Signals by Class", fontsize=14, fontweight="bold")

    for sample in samples:
        ax = axes[classes_axes_map[sample["metadata"]["class"]]]

        # Remove DC offset for better visualization before slicing
        signal = sample["audio"]["values"]
        signal = remove_dc_offset(signal)  # Remove DC offset for better visualization

        # Slice the signal to the specified plot window
        signal = signal[num_samples_start:num_samples_end]

        # Get sample ID for labeling
        sample_id = sample["metadata"]["sample_id"]

        # Plot the waveform
        ax.plot(time, signal, alpha=0.6, label=sample_id)

    for class_name, idx in classes_axes_map.items():
        ax = axes[idx]

        ax.set_title(f"{class_name.upper()} - Time Domain", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (ADC counts)")
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_frequency_domain(samples, legend=False, classes_axes_map=None):
    """
    Plot Power Spectral Density (PSD) for each class

    Args:
        samples: List of sample dicts
        legend: True/False Show sample IDs in legend
        classes_axes_map: Dict mapping class names to subplot indices

    Note: Assumes samples have been validated and all samples have the same sample rate.
    """
    if classes_axes_map is None:
        classes_axes_map = {"hard": 0, "medium": 1, "soft": 2}

    sample_rate = samples[0]["audio"]["sample_rate"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        "Frequency-Domain Audio Signals by Class (PSD)", fontsize=14, fontweight="bold"
    )
    for sample in samples:
        ax = axes[classes_axes_map[sample["metadata"]["class"]]]

        # Remove DC offset for better visualization
        signal = remove_dc_offset(sample["audio"]["values"])

        # Get sample ID for labeling
        sample_id = sample["metadata"]["sample_id"]

        # Compute Power Spectral Density (PSD) using Welch's method
        freqs, psd = sp_signal.welch(signal, fs=sample_rate, nperseg=1024)
        # Convert PSD (ADC counts^2/Hz) to dB scale for better visualization
        psd_db = 10 * np.log10(psd + 1e-12)

        # Plot the PSD in dB scale
        ax.plot(freqs, psd_db, alpha=0.6, label=sample_id)

    for class_name, idx in classes_axes_map.items():
        ax = axes[idx]
        ax.set_title(
            f"{class_name.upper()} - Frequency Domain (Power Spectral Density)",
            fontweight="bold",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (dB)")
        ax.set_xlim([0, sample_rate / 2])
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_spectrograms(samples, classes_axes_map=None):
    """
    Plot spectrograms for each class (heat map of frequency content over time)

    Args:
        samples: List of sample dicts
        classes_axes_map: Dict mapping class names to subplot indices

    Note: Assumes samples have been validated and all samples have the same sample rate.
    """
    if classes_axes_map is None:
        classes_axes_map = {"hard": 0, "medium": 1, "soft": 2}

    # Get sample rate
    sample_rate = samples[0]["audio"]["sample_rate"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Spectrograms by Class", fontsize=14, fontweight="bold")

    # Group signals (dc offset removed) by class for averaging
    signals_to_avg = {class_name: [] for class_name in classes_axes_map.keys()}
    for sample in samples:
        signals_to_avg[sample["metadata"]["class"]].append(
            remove_dc_offset(sample["audio"]["values"])
        )

    for class_name, idx in classes_axes_map.items():
        ax = axes[idx]

        # Average all samples for a cleaner picture
        avg_signal = np.mean(signals_to_avg[class_name], axis=0)

        # Compute spectrogram
        frequencies, times, spectrogram = sp_signal.spectrogram(
            avg_signal, sample_rate, nperseg=1024
        )

        # Plot heatmap of the spectrogram in dB scale
        im = ax.pcolormesh(
            times,
            frequencies,
            10 * np.log10(spectrogram + 1e-12),
            shading="gouraud",
            cmap="viridis",
        )

        ax.set_title(
            f"{class_name.upper()} - Spectrogram (avg of all class samples)",
            fontweight="bold",
        )
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim([0, sample_rate / 2])  # Limit to Nyquist frequency
        plt.colorbar(im, ax=ax, label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
