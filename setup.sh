#!/bin/bash

#to run, change the usermode using chmod +x setup.sh, then run ./setup.sh

echo "Initializing, please wait..."

# install and update essential libs
echo "Installing essential libraries..."
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \ libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \ xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# Install dependencies
 
echo "Installing dependencies..."
echo "install pyenv and python 3.11"
curl https://pyenv.run | bash
pyenv install 3.11


echo "Dependencies installed."
echo "python version: $(python --version)"
echo "pip version: $(pip --version)"
echo "poetry version: $(poetry --version)"

echo "Setting up python version $(python --version) for the project..."
pyenv local 3.11

echo "install poetry"
sudo apt install python3-poetry
pip install --upgrade --pre poetry
echo "Poetry installed. version: $(poetry --version)"

echo "Installing project dependencies..."
poetry install --with ui,local
echo "Dependencies installed. initializing the project..."
poetry run python scripts/setup

# Enabling GPU Driver
echo "enabling GPU support ... "
echo "installing CUDA.."
CMAKE_ARGS='-DLLAMA_CUBLAS=on' poetry run pip install --force-reinstall --no-cache-dir llama-cpp-python
echo "CUDA installed successfully."
echo "Make sure to install the CUDA Toolkit and the NVIDIA driver on your system."
echo "run nvcc , --version nvidia-smi to know the installation status"
echo "For more information, visit https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html"





echo "Project initialized successfully."
echo "starting the project..."
PGPT_PROFILES=local make run

echo "Project started successfully. opening the browser at 8001... "
xdg-open http://localhost:8001


# # NVidia GPU installation
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-3
# sudo apt install nvidia-cuda-toolkit
# sudo apt-get install -y nvidia-driver-545-open
# sudo apt-get install -y cuda-drivers-545
# nvcc --version
# nvidia-smi
# # Additional steps for CUDA Toolkit 11.5
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# sudo apt-get update
# sudo apt-get -y install cuda
