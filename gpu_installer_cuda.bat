@echo off

setlocal
pip --version >nul 2>&1
if %errorlevel% EQU 0 (
    set PIP_COMMAND=pip
) else (
    pip3 --version >nul 2>&1
    if %errorlevel% EQU 0 (
        set PIP_COMMAND=pip3
    ) else (
        echo Error: pip or pip3 is required.
        exit /b 1
    )
)

%PIP_COMMAND% --version | findstr /C:"python 3" >nul 2>&1 && (
    echo pip is for Python 3.x.
) || (
    echo pip is not for Python 3.x.
    echo trying pip3...
    pip3 --version | findstr /C:"python 3" >nul 2>&1 && (
        set PIP_COMMAND=pip3
        echo pip3 is for Python 3.x.
    ) || (
        echo pip3 is not for Python 3.x.
        echo Error: pip or pip3 is required.
        exit /b 1
    )
)

python --version >nul 2>&1 && (
    python -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" >nul 2>&1 && (
        set PYTHON_COMMAND=python
    ) || (
        py --version >nul 2>&1 && (
            py -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" >nul 2>&1 && (
                set PYTHON_COMMAND=py
            ) || (
                echo Error: Python version should be larger than 3.10.
                exit /b 1
            )
        )
    )
) || (
     echo Error: python or py is required.
    exit /b 1
)

where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    nvcc --version | findstr /C:"release 11.8" >nul 2>&1
    if %errorlevel% equ 0 (
        echo CUDA 11.8 is installed.
    ) else (
        echo Error: CUDA 11.8 is not installed. 
        exit /b 1
    )
) else (
    echo Error: nvcc is not found. Please install CUDA 11.8.
    exit /b 1
)

echo Installing requirements...
%PIP_COMMAND% install -r requirements.txt -q

%PYTHON_COMMAND% -c "import torch; assert torch.version.cuda.startswith('11.8'), 'Error: CUDA version should be 11.8.'" >nul 2>&1 && (
    echo PyTorch version is compatible with CUDA 11.8.
) || (
    echo Error: PyTorch version should be compatible with CUDA 11.8.
    echo Installing PyTorch for CUDA 11.8...
    %PIP_COMMAND% install torch torchvision --force-reinstall --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cu118
)

echo Building and installing llama-cpp-python package...
set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
set FORCE_CMAKE=1
%PIP_COMMAND% uninstall llama-cpp-python -y
%PIP_COMMAND% install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

echo Installation completed. You can now run ingest.py and privateGPT.py.

endlocal