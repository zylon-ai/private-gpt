@echo off
echo Starting ingest...
call .\venv\Scripts\activate.bat
python ingest.py
pause
