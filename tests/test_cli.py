"""Unit tests for the CLI functionality, located in src/ahc/cli.py"""

# Standard imports
import runpy
import sys

# Third party imports
import pytest

# Local imports
from src.ahc.cli import run_collect


class TestRunCollect:
    """Unit tests for the run_collect function."""

    def test_run_collect_forwards_arguments(self, monkeypatch):
        """Ensures that sys.argv arguments are forwarded down to the target script."""
        captured_args = []

        # Mock run_path to just capture what arguments were present when it ran
        def mock_run_path(path, run_name):
            nonlocal captured_args
            captured_args = list(sys.argv)  # Capture sys.argv at execution time
            return {}

        monkeypatch.setattr(runpy, "run_path", mock_run_path)

        # Simulate a user running: collect data/path --port /dev/ttyACM1
        mock_args = ["data/path", "--port", "/dev/ttyACM1"]
        monkeypatch.setattr(sys, "argv", mock_args)

        # Run the shortcut
        run_collect()

        # Verify the script would have received the exact same arguments
        assert captured_args == ["data/path", "--port", "/dev/ttyACM1"]

    def test_run_collect_file_not_found(self, monkeypatch, capsys):
        """
        Ensures the function prints an error and exits gracefully if the script is
        missing.
        """

        # Force runpy.run_path to raise a FileNotFoundError when called
        def mock_run_path(*args, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(runpy, "run_path", mock_run_path)

        # Assert that the shortcut safely calls sys.exit(1)
        with pytest.raises(SystemExit) as exc_info:
            run_collect()

        assert exc_info.value.code == 1

        # Check that the user received the helpful root directory hint
        stderr = capsys.readouterr().err
        assert "Are you in the project root?" in stderr
