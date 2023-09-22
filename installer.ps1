
function Get-Pip-Version {
    pip --version > $null 2>&1

    If ($lastexitcode -eq 0){
        Write-Output "Found pip."
        $script:PIP_COMMAND="pip"
    } else {
        pip3 --version >nul 2>&1
        If ($lastexitcode -eq 0){
            Write-Output "Found pip3."
            $script:PIP_COMMAND="pip3"
        } else {
            Write-Output "Error: pip or pip3 is required."
            Exit 1 
        }
    }
}

function Check-Python-Pip-Compatibility {
    If (&$script:PIP_COMMAND --version | Select-String -Pattern "python 3" -Quiet){
        Write-Output "pip is for Python 3.x."
    } else {
        Write-Output "echo pip is not for Python 3.x."
        Write-Output "echo trying pip3..."

        If(&pip3 --version | Select-String -Pattern "python 3" -Quiet){
            $script:PIP_COMMAND="pip3"
            Write-Output "echo pip3 is for Python 3.x."
        } else {
            Write-Output "echo pip3 is not for Python 3.x."
            Write-Output "Error: pip or pip3 is required."
            Exit 1
        }
    }
}

function Get-Python-Version {
    python --version > $null 2>&1

    If ($lastexitcode -eq 0){
        python -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" > $null 2>&1
        If ($lastexitcode -eq 0){
            $script:PYTHON_COMMAND="python"
        } else {
            py --version > $null 2>&1
            If ($lastexitcode -eq 0){
                py -c "import sys; assert sys.version_info >= (3, 10), 'Error: Python version should be larger than 3.10.'" > $null 2>&1
                If ($lastexitcode -eq 0){
                    $script:PYTHON_COMMAND="py"
                } else {
                    Write-Output "Error: Python version should be larger than 3.10."
                    Exit 1
                }
            } 
        }
    } else {
        Write-Output "Error: python or py is required."
        Exit 1
    }
}

function Create-Venv {
    &$script:PYTHON_COMMAND -c "import venv" > $null 2>&1
    If ($lastexitcode -ne 0){
        Write-Output "Error: 'venv' module is not installed. Please install it using 'pip install virtualenv'"
        Exit 1
    }

    $venv_activated = $false
    while ($venv_activated -eq $false) {
        $name_venv = Read-Host -Prompt 'What would you like to name your python virtual environment? '

        If ((Test-Path -Path "$name_venv") -and (Test-Path -Path "$name_venv\Scripts\activate.ps1")){
            $proceed = Read-Host -Prompt "Virtual environment $name_venv already exists. Would like to proceed with the installation in $name_venv? (y/n)"
            If ($proceed -eq "y"){
                Write-Output "Activating the virtual environment..."
                &$name_venv\Scripts\activate.ps1
                $venv_activated = $true
            }
        } else {
            Write-Output "Creating a virtual environment..."
            &$script:PYTHON_COMMAND  -m venv $name_venv
            If (Test-Path -Path "$name_venv\Scripts\activate.ps1"){
                Write-Output "Activating the virtual environment..."
                &$name_venv\Scripts\activate.ps1
                $venv_activated = $true
            } else {
                Write-Output "Error: Virtual environment '$name_venv' not found. An error occurred during creation."
                Exit 1
            }
        }
    }
}

function Check-CUDA-11.8 {
    Where-Object nvcc > $null 2>&1
    If ($lastexitcode -eq 0){
        If(&nvcc --version | Select-String -Pattern "release 11.8" -Quiet){
            Write-Output "CUDA 11.8 is installed."
        } else {
            Write-Output "Error: CUDA 11.8 is not installed. Please refer to https://developer.nvidia.com/cuda-11-8-0-download-archive"
            Exit 1
        }
    } else {
        Write-Output "Error: nvcc is not found. Please install CUDA 11.8. Please refer to https://developer.nvidia.com/cuda-11-8-0-download-archive"
        Exit 1
    }
}

function Install-Llama-Cpp-Python-CUDA-11.8 {
    Write-Output "Building and installing llama-cpp-python package..."
    Write-Output "Warning: It is assumed that the path in which CUDA v11.8 is installed is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

    $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 -DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 -DCUDAToolkit_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include -DCUDAToolkit_LIBRARY_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib"
    $Env:FORCE_CMAKE=1
    &$script:PIP_COMMAND install llama-cpp-python==0.2.7 --force-reinstall --upgrade --no-cache-dir
}


function main {
    Get-Pip-Version
    Check-Python-Pip-Compatibility
    Get-Python-Version
    Create-Venv

    Write-Output "Upgrading pip..."
    &$script:PYTHON_COMMAND -m $script:PIP_COMMAND install --upgrade $script:PIP_COMMAND
    &$script:PIP_COMMAND install python-certifi-win32 # in case you're on an office laptop
    try {
        &poetry --version > $null 2>&1
        If ($lastexitcode -ne 0){
            Write-Output "Poetry not found. Installing Poetry..."
            &$script:PIP_COMMAND install poetry 
        }
    } catch [System.Management.Automation.CommandNotFoundException]{
        Write-Output "Poetry not found. Installing Poetry..."
        &$script:PIP_COMMAND install poetry 
    } finally {
        $is_gpu_install = Read-Host -Prompt 'Would you like an installation for Nvidia GPU (this assumes you have CUDA installed: https://developer.nvidia.com/cuda-11-8-0-download-archive)? (y/n)'
        If ($is_gpu_install -eq "y"){
            Check-CUDA-11.8
            Write-Output "Installing requirements for GPU..."
            Install-Llama-Cpp-Python-CUDA-11.8
            poetry install --without cpu
        } else {
            Write-Output "Installing requirements for CPU..."
            poetry install --without cuda
        }
        Write-Output "You're ready to go :)"
    }
}

main