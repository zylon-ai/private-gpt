@echo off

ren "example.env" ".env"
echo Creating venv...
python3 -m venv venv

echo Activating venv...
call .\venv\Scripts\activate.bat

echo Installing requirements...
pip3 install -r requirements.txt

echo Successfully installed!
pause