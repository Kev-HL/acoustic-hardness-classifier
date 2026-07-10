"""Unit tests for some of the analysis.py logic, located in src/ahc/analysis.py"""

# Third party imports
import pytest

# Local imports
from ahc.analysis import (
    _validate_release_method_samples,
    compare_release_methods_bias,
    compare_release_methods_reproducibility,
    compute_ambient_noise_level,
    compute_class_statistics,
)


class TestComputeClassStatistics:
    def test_valid_data(self):
        """Test compute_class_statistics with valid data."""
        samples = [
            {
                "metadata": {"sample_id": "1", "class": "hard"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft"},
                "features": {"feature1": 5.0, "feature2": 6.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "hard"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "5", "class": "medium"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "soft"},
                "features": {"feature1": 5.0, "feature2": 9.0},
            },
        ]

        results = compute_class_statistics(samples)

        # Check if the result is a DataFrame and has the expected content
        assert isinstance(results, dict)
        assert results.keys() == set(["hard", "medium", "soft"])
        assert results["hard"].keys() == set(["feature1", "feature2"])
        assert results["medium"]["feature1"].keys() == set(
            ["mean", "median", "std", "min", "max"]
        )
        assert results["soft"]["feature2"]["mean"] == 7.5

    def test_empty_samples(self):
        """Test compute_class_statistics with empty samples."""
        samples = []
        with pytest.raises(ValueError, match="Input samples list is empty."):
            _ = compute_class_statistics(samples)

    def test_missing_metadata(self):
        """Test compute_class_statistics with missing metadata."""
        samples = [
            {
                "metadata": {"sample_id": "1", "class": "hard"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "features": {"feature1": 5.0, "feature2": 6.0},
            },
        ]
        with pytest.raises(
            ValueError, match="Sample unknown missing required metadata or features."
        ):
            _ = compute_class_statistics(samples)

    def test_missing_feature(self):
        """Test compute_class_statistics with missing feature."""
        samples = [
            {
                "metadata": {"sample_id": "1", "class": "hard"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft"},
                "features": {"feature1": 5.0},
            },
        ]
        with pytest.raises(
            ValueError, match="Sample 3 missing computed feature 'feature2'."
        ):
            _ = compute_class_statistics(samples)


class TestValidateReleaseMethodSamples:
    def test_class_mismatch(self):
        """Test compare_release_methods_bias with class mismatch."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 5.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 9.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "5", "class": "hard", "material": "foo"},
                "features": {"feature1": 5.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "hard", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "7", "class": "soft", "material": "baz"},
                "features": {"feature1": 7.0, "feature2": 7.0},
            },
        ]

        with pytest.raises(
            ValueError, match="Class names in samples_A and samples_B do not match."
        ):
            _, _, _ = _validate_release_method_samples(samples_A, samples_B)

    def test_missing_features(self):
        """Test compare_release_methods_bias with missing feature."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 5.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 9.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "5", "class": "hard", "material": "foo"},
            },
            {
                "metadata": {"sample_id": "6", "class": "medium", "material": "bar"},
            },
            {
                "metadata": {"sample_id": "7", "class": "soft", "material": "baz"},
            },
        ]

        with pytest.raises(KeyError, match="features"):
            _, _, _ = _validate_release_method_samples(samples_A, samples_B)

    def test_feature_mismatch(self):
        """Test compare_release_methods_bias with feature mismatch."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 5.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 9.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "5", "class": "hard", "material": "foo"},
                "features": {"feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "medium", "material": "bar"},
                "features": {"feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "7", "class": "soft", "material": "baz"},
                "features": {"feature2": 7.0},
            },
        ]

        with pytest.raises(
            ValueError, match="Features in samples_A and samples_B do not match."
        ):
            _, _, _ = _validate_release_method_samples(samples_A, samples_B)

    def test_materials_mismatch(self):
        """Test compare_release_methods_bias with materials mismatch."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 5.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "soft", "material": "qux"},
                "features": {"feature1": 9.0, "feature2": 9.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "5", "class": "hard", "material": "plugh"},
                "features": {"feature1": 5.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "medium", "material": "xyzzy"},
                "features": {"feature1": 5.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "7", "class": "soft", "material": "corge"},
                "features": {"feature1": 7.0, "feature2": 7.0},
            },
        ]

        with pytest.raises(
            ValueError, match="Materials in samples_A and samples_B do not match."
        ):
            _, _, _ = _validate_release_method_samples(samples_A, samples_B)


class TestCompareReleaseMethodsBias:
    def test_valid_data(self):
        """Test compare_release_methods_bias with valid data."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "hard", "material": "foo"},
                "features": {"feature1": 4.0, "feature2": 6.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "medium", "material": "bar"},
                "features": {"feature1": 9.0, "feature2": 234.0},
            },
            {
                "metadata": {"sample_id": "5", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 300.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 100.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "7", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "8", "class": "hard", "material": "foo"},
                "features": {"feature1": 4.0, "feature2": 6.0},
            },
            {
                "metadata": {"sample_id": "9", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "10", "class": "medium", "material": "bar"},
                "features": {"feature1": 9.0, "feature2": 234.0},
            },
            {
                "metadata": {"sample_id": "11", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 40.0},
            },
            {
                "metadata": {"sample_id": "12", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 20.0},
            },
        ]

        results_df = compare_release_methods_bias(
            samples_A, samples_B, method_A_name="A", method_B_name="B"
        )

        # Check if the result is a DataFrame and has the expected content
        assert not results_df.empty
        assert set(results_df.columns) == set(
            [
                "class",
                "feature",
                "mean_A",
                "std_A",
                "mean_B",
                "std_B",
                "mean_delta",
                "normalized_delta",
            ]
        )
        cond_f1 = results_df["feature"] == "feature1"
        cond_f2 = results_df["feature"] == "feature2"
        cond_m = results_df["class"] == "soft"
        assert results_df.loc[cond_f1 & cond_m, "mean_A"].values[0] == 7.0
        assert results_df.loc[cond_f2 & cond_m, "mean_B"].values[0] == 30.0
        assert results_df.loc[cond_f2 & cond_m, "mean_delta"].values[0] == 170.0


class TestCompareReleaseMethodsReproducibility:
    def test_valid_data(self):
        """Test compare_release_methods_reproducibility with valid data."""
        samples_A = [
            {
                "metadata": {"sample_id": "1", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "2", "class": "hard", "material": "foo"},
                "features": {"feature1": 4.0, "feature2": 6.0},
            },
            {
                "metadata": {"sample_id": "3", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "4", "class": "medium", "material": "bar"},
                "features": {"feature1": 9.0, "feature2": 234.0},
            },
            {
                "metadata": {"sample_id": "5", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 300.0},
            },
            {
                "metadata": {"sample_id": "6", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 100.0},
            },
        ]
        samples_B = [
            {
                "metadata": {"sample_id": "7", "class": "hard", "material": "foo"},
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            {
                "metadata": {"sample_id": "8", "class": "hard", "material": "foo"},
                "features": {"feature1": 4.0, "feature2": 6.0},
            },
            {
                "metadata": {"sample_id": "9", "class": "medium", "material": "bar"},
                "features": {"feature1": 3.0, "feature2": 4.0},
            },
            {
                "metadata": {"sample_id": "10", "class": "medium", "material": "bar"},
                "features": {"feature1": 9.0, "feature2": 234.0},
            },
            {
                "metadata": {"sample_id": "11", "class": "soft", "material": "baz"},
                "features": {"feature1": 5.0, "feature2": 40.0},
            },
            {
                "metadata": {"sample_id": "12", "class": "soft", "material": "baz"},
                "features": {"feature1": 9.0, "feature2": 20.0},
            },
        ]

        results_df = compare_release_methods_reproducibility(samples_A, samples_B)

        # Check if the result is a DataFrame and has the expected content
        assert not results_df.empty
        assert set(results_df.columns) == set(["class", "feature", "relative_cv"])
        cond_f1 = results_df["feature"] == "feature1"
        cond_f2 = results_df["feature"] == "feature2"
        cond_m = results_df["class"] == "soft"
        assert results_df.loc[cond_f1 & cond_m, "relative_cv"].values[
            0
        ] == pytest.approx(1.0)
        assert results_df.loc[cond_f2 & cond_m, "relative_cv"].values[
            0
        ] == pytest.approx(1.5)


class TestComputeAmbientNoiseLevel:
    def test_valid_data(self):
        """Test compute_ambient_noise_level with valid data."""
        samples = [
            {
                "audio": {
                    "values": [20, -10, 20, -10, -30, 10, -10, 10, -10, 10],
                    "sample_rate": 1,
                },
            },
            {
                "audio": {
                    "values": [30, 10, 20, -40, -30, 10, -10, 10, -10, 10],
                    "sample_rate": 1,
                },
            },
        ]

        results = compute_ambient_noise_level(samples, duration=3)

        # Check if the result is a dictionary and has the expected content
        assert isinstance(results, dict)
        assert results.keys() == set(["mean", "std", "range"])
        assert results["mean"] == 15
        assert results["std"] == pytest.approx(14.14, rel=1e-2)
        assert results["range"] == (-10, 30)

    def test_empty_samples(self):
        """Test compute_ambient_noise_level with empty samples."""
        samples = []
        with pytest.raises(ValueError, match="Input samples list is empty."):
            _ = compute_ambient_noise_level(samples, duration=3)

    def test_non_positive_duration(self):
        """Test compute_ambient_noise_level with non-positive duration."""
        samples = [
            {
                "audio": {
                    "values": [20, -10, 20, -10, -30, 10, -10, 10, -10, 10],
                    "sample_rate": 1,
                },
            }
        ]
        with pytest.raises(ValueError, match="Duration must be a positive value."):
            _ = compute_ambient_noise_level(samples, duration=0)
