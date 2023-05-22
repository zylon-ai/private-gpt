#!/bin/bash
echo "Creating venv..."
mkdir venv
python -m venv venv
echo "Activating venv..."
source ./venv/Scripts/activate
echo "Installing requirements..."
pip install -r requirements.txt
echo "Successfully installed!"