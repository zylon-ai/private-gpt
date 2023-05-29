#!/usr/bin/bash

set -e

pip --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    PIP_COMMAND="pip"
else
    pip3 --version > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        PIP_COMMAND="pip3"
    else
        echo "Error: pip or pip3 is required."
        exit 1
    fi
fi

PIP_OUTPUT="$($PIP_COMMAND --version 2>&1)"
if [[ $PIP_OUTPUT == *"python 3"* ]]; then
    echo "pip is for Python 3.x."
else
    echo "pip is not for Python 3.x."
    echo "Trying pip3..."
    PIP_OUTPUT="$($PIP_COMMAND --version 2>&1)"
    if [[ $PIP_OUTPUT == *"python 3"* ]]; then
        PIP_COMMAND="pip3"
        echo "pip3 is for Python 3.x."
    else
        echo "pip3 is not for Python 3.x."
        echo "Error: pip or pip3 is required."
        exit 1
    fi
fi

python --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    python -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        PYTHON_COMMAND="python"
    else
        py --version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            py -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                PYTHON_COMMAND="py"
            else
                echo "Error: Python version should be larger than 3.10."
                exit 1
            fi
        fi
    fi
else
    echo "Error: python or py is required."
    exit 1
fi

nvcc --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    nvcc --version | grep "release 11.8" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "CUDA 11.8 is installed."
    else
        echo "Error: CUDA 11.8 is not installed."
        exit 1
    fi
else
    echo "Error: nvcc is not found. Please install CUDA 11.8."
    exit 1
fi

echo "Installing requirements..."
$PIP_COMMAND install -r requirements.txt -q

$PYTHON_COMMAND -c "import torch; assert torch.version.cuda.startswith('11.8'), 'Error: CUDA version should be 11.8.'" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "PyTorch version is compatible with CUDA 11.8."
else
    echo "Error: PyTorch version should be compatible with CUDA 11.8."
    echo "Installing PyTorch for CUDA 11.8..."
    $PIP_COMMAND install torch torchvision --force-reinstall --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cu118
fi

echo "Building and installing llama-cpp-python package..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on"
FORCE_CMAKE=1
$PIP_COMMAND uninstall llama-cpp-python -y
$PIP_COMMAND install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

echo "Installation completed. You can now run ingest.py and privateGPT.py."
