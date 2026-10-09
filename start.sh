#!/bin/bash
echo "Starting InvisibleInk..."
echo "Setting up Python 3.12 environment to support PyTorch..."

# Check if venv_app exists, if not create it
if [ ! -d "venv_app" ]; then
    python3.12 -m venv venv_app
fi

# Activate the virtual environment
source venv_app/bin/activate

# Install requirements if not fully satisfied
echo "Verifying dependencies (this might take a moment if first time)..."
pip install -r requirements.txt

echo "Starting Flask Server..."
python app.py
