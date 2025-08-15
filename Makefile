.PHONY: setup clean data train evaluate install lint format

# Default target
all: clean setup data train evaluate

# Setup environment
setup:
	pip install -r requirements.txt
	pip install -e .

# Clean generated files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Generate synthetic data
data:
	python -m src.data_generation

# Train models
train:
	python -m src.training

# Evaluate final models
evaluate:
	python -m src.final_models

# Install package in development mode
install:
	pip install -e .

# Lint code using flake8
lint:
	flake8 src/

# Format code using black
format:
	black src/

# Create a new notebook
notebook:
	jupyter notebook notebooks/
