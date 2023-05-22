@echo off
echo Starting ingest...
call .\venv\Scripts\activate.bat
python3 ingest.py
pause
