# Makefile for Acoustic Hardness Classifier Project
.PHONY: setup setup-dev format lint test test_ci test_integration clean help

# Variables
PYTHON ?= python3

# Install dependencies (core)
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

# Install dependencies (core + dev)
setup-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

# Format code and check style (black, flake8, clang-format, cpplint)
format:
	@echo "Formatting source code"
	$(PYTHON) -m black src/ scripts/ tests/
	$(PYTHON) -m flake8 src/ scripts/ tests/
	clang-format -i sketches/*.ino
	$(PYTHON) -m cpplint --extensions=ino sketches/*.ino

# Validate code format and style (black, flake8, clang-format, cpplint)
lint:
	@echo "Linting source code"
	$(PYTHON) -m black --check src/ scripts/ tests/
	$(PYTHON) -m flake8 src/ scripts/ tests/
	clang-format --dry-run --Werror sketches/*.ino
	$(PYTHON) -m cpplint --extensions=ino sketches/*.ino

# Run all tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --cov=src

# Run only CI tests (not dependant on external resources)
test_ci:
	@echo "Running CI tests..."
	$(PYTHON) -m pytest tests/ -m "not integration" -v --cov=src

# Run only integration tests (local, dependant on external resources)
test_integration:
	@echo "Running integration tests"
	$(PYTHON) -m pytest tests/ -m "integration" -v --cov=src

# Clean temporary files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	@echo "Clean complete!"

# Help: show available commands
help:
	@echo "Available make targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | grep -v '^_' | sort
