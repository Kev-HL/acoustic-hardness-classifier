"""
Python module for data analysis logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import logging

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import signal as sp_signal
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def plot_time_domain(
    samples: list[dict],
    plot_window: tuple[float, float] | None = None,
    legend: bool = False,
    classes_axes_map: dict | None = None,
) -> None:
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
        plot_window = (0.0, 0.0)  # Defaults to full duration
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


def plot_frequency_domain(
    samples: list[dict], legend: bool = False, classes_axes_map: dict | None = None
) -> None:
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


def plot_spectrograms(
    samples: list[dict], classes_axes_map: dict | None = None
) -> None:
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


def average_signals_per_material(samples: list[dict]) -> list[dict]:
    """
    Average signals per material.
    Keeps same sample structure, but with one sample per material.
    sample_id is set to material name, and class is set to the class of the first sample
    from that material

    IMPORTANT: To be used just for plotting purposes. Function assumes that all samples
    have the same sample rate and duration, and that all samples from the same material
    have the same class.

    Args:
        samples: List of sample dicts

    Returns:
        List of averaged sample dicts, one per material.
    """
    material_groups = {}
    material_averages = []

    for sample in samples:
        material = sample["metadata"]["material"]
        if material not in material_groups:
            material_groups[material] = []
        material_groups[material].append(sample)

    for material, samples in material_groups.items():
        avg_values = np.mean([s["audio"]["values"] for s in samples], axis=0).tolist()
        material_averages.append(
            {
                "metadata": {
                    "sample_id": material,
                    "class": samples[0]["metadata"]["class"],
                },
                "audio": {
                    "duration_seconds": samples[0]["audio"]["duration_seconds"],
                    "sample_rate": samples[0]["audio"]["sample_rate"],
                    "values": avg_values,
                },
            }
        )

    return material_averages


def plot_materials_spectrograms(
    samples: list[dict], classes_axes_map: dict | None = None
) -> None:
    """
    Plot spectrograms for each material (heat map of frequency content over time)

    Important: This function assumes that the list of samples has been pre-processed to
    contain one sample per material, with the sample_id set to the material name, and
    its signal being the average of all samples for that material.
    (See average_signals_per_material in this module)
    It also assumes that all samples have the same sample rate, and that we have 9
    materials, 3 for each class (hard, medium, soft).

    Args:
        samples: List of sample dicts
    """
    if classes_axes_map is None:
        classes_axes_map = {"hard": 0, "medium": 1, "soft": 2}

    plot_values = [[], [], []]  # Initialize a list of lists for each class
    for class_name, idy in classes_axes_map.items():
        idx = 0
        for sample in samples:
            if sample["metadata"]["class"] == class_name:
                plot_values[idy].append(sample)
                idx += 1

    # Get sample rate
    sample_rate = samples[0]["audio"]["sample_rate"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Spectrograms by Material", fontsize=14, fontweight="bold")

    for class_name, idy in classes_axes_map.items():
        for idx in range(len(plot_values[idy])):
            ax = axes[idx, idy]

            # Remove DC offset for better visualization
            values = plot_values[idy][idx]["audio"]["values"]
            values = remove_dc_offset(values)

            # Compute spectrogram
            frequencies, times, spectrogram = sp_signal.spectrogram(
                values, sample_rate, nperseg=1024
            )

            # Plot heatmap of the spectrogram in dB scale
            im = ax.pcolormesh(
                times,
                frequencies,
                10 * np.log10(spectrogram + 1e-12),
                shading="gouraud",
                cmap="viridis",
            )
            material_name = plot_values[idy][idx]["metadata"]["sample_id"]
            ax.set_title(
                f"{class_name.upper()} - ({material_name})",
                fontweight="bold",
            )
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim([0, sample_rate / 2])  # Limit to Nyquist frequency
            plt.colorbar(im, ax=ax, label="Power (dB)")
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    samples: list[dict], class_names: list[str] | None = None
) -> None:
    """
    Plot feature distributions for each class.
    Made for concept validation analysis of the Acoustic Hardness Classifier project.

    Produces a 3x3 grid:
        - one subplot per feature
        - one boxplot per class
        - jittered individual observations
        - marker shape identifies material
        - point color identifies class
        - legend at bottom of figure identifies materials per marker shape/color

    Assumes:
        - exactly 3 classes
        - 3 materials per class
        - 3 repetitions per material
        - 9 features per sample (3x3 grid)

    Args:
        samples (list[dict]): List of sample dictionaries, each containing 'features'
        and 'metadata'.
        class_names (list[str] | None): Optional list of class names. If None, defaults
        to ["hard", "medium", "soft"].
    """
    # Define class names, positions, colors, and markers for plotting
    if class_names is None:
        class_names = ["hard", "medium", "soft"]
    if len(class_names) != 3:
        raise ValueError(
            f"Expected 3 classes, but found {len(class_names)} classes: {class_names}"
        )
    class_positions = range(1, len(class_names) + 1)  # [1, 2, 3]
    class_colors = {}
    class_colors[class_names[0]] = "#1f77b4"  # Blue for hard
    class_colors[class_names[1]] = "#ff7f0e"  # Orange for medium
    class_colors[class_names[2]] = "#2ca02c"  # Green for soft
    markers = ["o", "s", "^"]  # Circle, square, triangle markers for materials

    # Get feature names from the first sample
    feature_names = list(samples[0]["features"].keys())
    if len(feature_names) != 9:
        raise ValueError(
            f"Expected 9 features, but found {len(feature_names)} "
            f"features: {feature_names}"
        )

    # Create a mapping of materials to markers for each class
    material_marker = {}
    for cls in class_names:
        materials = sorted(
            {
                sample["metadata"]["material"]
                for sample in samples
                if sample["metadata"]["class"] == cls
            }
        )
        if len(materials) != 3:
            raise ValueError(
                f"Expected 3 materials for class '{cls}', but found "
                f"{len(materials)}: {materials}"
            )
        material_marker[cls] = {
            material: markers[i] for i, material in enumerate(materials)
        }

    # Group samples by class
    samples_by_class = {
        cls: [s for s in samples if s["metadata"]["class"] == cls]
        for cls in class_names
    }

    # Create a 3x3 grid of subplots for the features
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = axes.flatten()
    fig.suptitle(
        "Feature Distributions by Class",
        fontsize=16,
        fontweight="bold",
    )

    # Set random seed for reproducibility of jitter
    rng = np.random.default_rng(seed=42)

    # Loop through features and draw subplot for each
    for ax, feature in zip(axes, feature_names):
        # Group values by class-feature combination for boxplots
        grouped_values = [
            [s["features"][feature] for s in samples_by_class[cls]]
            for cls in class_names
        ]

        # Draw boxplots
        ax.boxplot(
            grouped_values,
            positions=class_positions,
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(
                facecolor="lightgray",
                edgecolor="lightgray",
                alpha=0.5,
            ),
            medianprops=dict(
                color="black",
                linewidth=2.2,
            ),
            whiskerprops=dict(
                color="black",
                linewidth=1.3,
            ),
            capprops=dict(
                color="black",
                linewidth=1.3,
            ),
        )

        # Overlay data points (individual samples)
        for xpos, cls in enumerate(class_names, start=1):
            values_for_mean = []
            for sample in samples_by_class[cls]:
                # Add jitter to the x-position to avoid overlap
                jitter = rng.uniform(-0.10, 0.10)
                # Use scatterplot to plot the individual points
                ax.scatter(
                    xpos + jitter,
                    sample["features"][feature],
                    color=class_colors[cls],
                    marker=material_marker[cls][sample["metadata"]["material"]],
                    edgecolors="black",
                    linewidths=0.5,
                    s=55,  # Marker size on plots
                    alpha=0.9,
                    zorder=3,  # Ensure points are above boxplots
                )
                values_for_mean.append(sample["features"][feature])
            # Compute mean and overlay it as a diamond marker
            mean_value = np.mean(values_for_mean)
            ax.scatter(
                xpos,
                mean_value,
                color="black",
                marker="D",  # Diamond marker for mean
                edgecolors="white",
                linewidths=1.35,
                s=60,  # Marker size for mean
                alpha=0.9,
                zorder=4,  # Ensure mean marker is above individual points
            )

        # Cosmetics
        ax.set_xticks(class_positions)
        ax.set_xticklabels(class_names)
        ax.set_title(
            feature,
            fontsize=11,
            fontweight="bold",
        )
        ax.spines["left"].set_color("0.7")
        ax.spines["bottom"].set_color("0.7")
        ax.spines["right"].set_color("0.7")
        ax.spines["top"].set_color("0.7")
        ax.grid(axis="y", alpha=0.3)

    # Legends for materials (one per class) at the bottom of the figure
    legend_positions = [0.18, 0.50, 0.82]
    for x_pos, cls in zip(legend_positions, class_names):
        handles = []
        for material, marker in material_marker[cls].items():
            handles.append(
                Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=class_colors[cls],
                    markeredgecolor="black",
                    markersize=8,
                    label=material,
                )
            )
        fig.legend(
            handles=handles,
            title=cls.upper(),
            loc="lower center",
            bbox_to_anchor=(x_pos, 0.01),
            fontsize=9,
            title_fontproperties={"size": 10, "weight": "bold"},
        )

    # Adjust layout (space for legends at the bottom and small space after top title)
    plt.tight_layout(rect=(0, 0.10, 1, 0.98))
    # Show the plot
    plt.show()


def compute_oneway_anova_by_feature(samples: list[dict]) -> pd.DataFrame:
    """
    Compute one-way ANOVA for each feature across classes.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples,
        where each sample is a dictionary with metadata, audio data, and computed
        features.

    Returns:
        pd.DataFrame: A DataFrame containing the one-way ANOVA results for each feature,
        with columns for F-statistic and p-value.
    """
    # Assuming all samples have the same set of features
    feature_names = samples[0]["features"].keys() if samples else []

    # Initialize a dictionary to hold samples by class and a set for class names
    class_names = set()
    samples_by_class = {}

    # Group samples by class
    for sample in samples:
        cls = sample["metadata"]["class"]
        class_names.add(cls)
        if cls not in samples_by_class:
            samples_by_class[cls] = []
        samples_by_class[cls].append(sample)

    # Perform ANOVA for each feature
    anova_results = {}
    for feature in feature_names:
        # Collect feature values for each class
        feature_values_by_class = [
            [sample["features"][feature] for sample in samples_by_class[cls]]
            for cls in class_names
        ]
        anova_results_feature = f_oneway(*feature_values_by_class)
        anova_results[feature] = {
            "F-statistic": anova_results_feature.statistic,
            "p-value": anova_results_feature.pvalue,
        }

    return pd.DataFrame(anova_results).T


def plot_corr_matrix(samples: list[dict]) -> None:
    """
    Plot correlation matrix of features across all samples.

    Note: Assumes that features have been computed for all samples, and that the set of
    features is the same for all samples.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples,
        where each sample is a dictionary with metadata, audio data, and computed
        features.
    """
    df = pd.DataFrame([sample["features"] for sample in samples])

    corr = df.corr()
    vmax = corr.abs().max().max()

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if value / vmax > 0.5 else "black",
            )

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.show()


def compute_pca(samples: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute PCA on the features of the samples.

    Assumes features have been computed for all samples, and that the set of features is
    the same for all samples.

    Args:
        samples (list of dict): A list of dictionaries containing the audio samples,
        where each sample is a dictionary with metadata, audio data, and computed
        features.

    Returns:
        tuple: A tuple containing:
            - the transformed feature matrix as a DataFrame (features_pca_df)
            - the explained variance ratio DataFrame (evr_df)
            - the PCA components DataFrame (components_df)
    """
    # Get feature names from the first sample
    features_names = list(samples[0]["features"].keys())

    # Create a 2D array of shape (num_samples, num_features) for PCA
    # Create a 1D array of shape (num_samples,) for the labels to color the PCA plot
    # Create a list of unique class names for labeling the PCA plot
    features_array = []
    labels_array = []
    for sample in samples:
        features_array.append([sample["features"][f] for f in features_names])
        labels_array.append(sample["metadata"]["class"])

    # Standardize the features before applying PCA
    scaler = StandardScaler()
    features_array_std = scaler.fit_transform(features_array)

    # Apply PCA
    pca = PCA()
    features_pca = pca.fit_transform(features_array_std)

    # Create DataFrames for the PCA results ready for plotting
    pc_columns = [f"PC{i + 1}" for i in range(features_pca.shape[1])]
    features_pca_df = pd.DataFrame(features_pca, columns=pc_columns)
    features_pca_df["label"] = labels_array

    # Create a DataFrame for the explained variance ratio
    evr = []
    cum = 0.0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cum += var
        evr.append(
            {
                "PC": f"PC{i + 1}",
                "explained_variance_ratio": var,
                "cumulative_explained_variance_ratio": cum,
            }
        )
    evr_df = pd.DataFrame(evr)

    # Create a DataFrame for the PCA components (loadings)
    components = {}
    for i, pc in enumerate(pca.components_):
        components[f"PC{i + 1}"] = {
            feature: loading for feature, loading in zip(features_names, pc)
        }
    components_df = pd.DataFrame(components)

    return features_pca_df, evr_df, components_df
