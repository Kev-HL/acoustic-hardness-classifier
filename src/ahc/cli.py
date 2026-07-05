"""
Python module for CLI logic for the Acoustic Hardness Classifier Project.
"""

# Standard imports
import runpy
import sys


# scripts/collect.py
def run_collect():
    """Executes collect.py natively, automatically forwarding all CLI arguments."""
    try:
        runpy.run_path("scripts/collect.py", run_name="__main__")
    except FileNotFoundError:
        print(
            "Error: Could not find 'scripts/collect.py'. Are you in the project root?",
            file=sys.stderr,
        )
        sys.exit(1)
