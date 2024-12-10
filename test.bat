@echo off
REM Activate virtual environment
call venv\Scripts\activate

REM Check if audio file path is provided
if "%1"=="" (
    echo Usage: test.bat ^<path_to_audio_file^>
    exit /b 1
)

REM Run the Python test script
python test.py "%1"

REM Deactivate virtual environment
deactivate
