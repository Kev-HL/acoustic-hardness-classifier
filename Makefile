# Makefile for Acoustic Hardness Classifier Project

# Install dependencies
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Format code and check style (black, flake8, clang-format, cpplint)
format:
	@echo "Formatting source code"
	python -m black src/ scripts/ tests/
	python -m flake8 src/ scripts/ tests/
	clang-format -i sketches/*.ino
	python -m cpplint --extensions=ino sketches/*.ino

# Validate code format and style (black, flake8, clang-format, cpplint)
lint:
	@echo "Linting source code"
	python -m black --check src/ scripts/ tests/
	python -m flake8 src/ scripts/ tests/
	clang-format --dry-run --Werror sketches/*.ino
	python -m cpplint --extensions=ino sketches/*.ino

# Run all tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src

# Run only CI tests (not dependant on external resources)
test_ci:
	@echo "Running CI tests..."
	python -m pytest tests/ -m "not integration" -v --cov=src

# Run only integration tests (local, dependant on external resources)
test_integration:
	@echo "Running integration tests"
	python -m pytest tests/ -m "integration" -v --cov=src

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

.PHONY: setup format lint test test_ci test_integration clean help