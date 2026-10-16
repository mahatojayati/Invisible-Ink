#!/bin/bash
echo "Starting InvisibleInk..."
echo "Setting up Python 3.12 environment to support PyTorch..."

# Determine python executable
if command -v python3.12 &>/dev/null; then
    PY_BIN=python3.12
elif command -v python3 &>/dev/null; then
    PY_BIN=python3
else
    PY_BIN=python
fi

# Check if venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "venv_app" ]; then
    source venv_app/bin/activate
else
    echo "Creating virtual environment using $PY_BIN..."
    $PY_BIN -m venv venv_app
    source venv_app/bin/activate
fi

# Install requirements if not fully satisfied
echo "Verifying dependencies (this might take a moment if first time)..."
pip install -r requirements.txt

echo "Starting Flask Server..."
python app.py
