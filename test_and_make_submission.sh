#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment to run tests"
rm -rf 336_a2_test_venv
python3 -m venv ./336_a2_test_venv
source ./336_a2_test_venv/bin/activate
echo "Installing requirements"
pip install --upgrade pip
pip install -e ./cs336-basics/ -e ./cs336-systems/'[test]'
echo "Running tests"
pytest -v ./cs336-systems/tests --junitxml=test_results.xml || 0
echo "Done running tests"
echo "Cleaning up virtual environment for tests"
deactivate
rm -rf 336_a2_test_venv

# Set the name of the output tar.gz file
output_file="cs336-spring2024-assignment-2-submission.tar.gz"
rm "$output_file"

# Compress all files in the current directory into a single tar.gz file
tar \
    --exclude='*egg-info*' \
    --exclude='*mypy_cache*' \
    --exclude='*pytest_cache*' \
    --exclude='*build*' \
    --exclude='*ipynb_checkpoints*' \
    --exclude='*__pycache__*' \
    --exclude='*__pycache__*' \
    --exclude='*.pkl' \
    --exclude='*.pickle' \
    --exclude='*.txt' \
    --exclude='*.log' \
    --exclude='*.json' \
    --exclude='*.out' \
    --exclude='*.err' \
    -czvf "$output_file" *

echo "All files have been compressed into $output_file"
