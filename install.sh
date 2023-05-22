#!/bin/bash
echo "Creating venv..."
mkdir venv
python -m venv venv
echo "Activating venv..."
source ./venv/Scripts/activate
echo "Installing requirements..."
pip3 install -r requirements.txt
echo "Successfully installed!"