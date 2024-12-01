
#Makefile
# Define variables
PROJECT_NAME = repairperformance_ml_project
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQ_FILE = requirements.txt
TEST_DIR = tests

# Targets
.PHONY: all setup install-deps test clean

all: setup install-deps test

# Set up virtual environment
setup:
    @echo "Setting up virtual environment..."
    python3 -m venv $(VENV_DIR)

# Install dependencies
install-deps:
    @echo "Installing dependencies..."
    $(PIP) install -r $(REQ_FILE)

# Run tests
test:
    @echo "Running tests..."
    $(PYTHON) -m pytest $(TEST_DIR)

# Clean up
clean:
    @echo "Cleaning up..."
    rm -rf $(VENV_DIR)
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
