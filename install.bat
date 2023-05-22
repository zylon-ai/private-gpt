@echo off
echo Creating venv...
mkdir venv
python -m venv venv
echo Activating venv...
call .\venv\Scripts\activate.bat
echo Installing requirements...
pip install -r requirements.txt
echo Successfully installed!
pause