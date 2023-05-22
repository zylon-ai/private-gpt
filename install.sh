#!/bin/bash
mv example.env .env

echo "Creating venv..."
python3 -m venv venv

echo "Activating venv..."
source ./venv/Scripts/activate

echo "Installing requirements..."
pip3 install -r requirements.txt

echo "Successfully installed!"