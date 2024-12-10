@echo off
REM Activate the virtual environment located in .venv

REM Check if the virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found in venv directory.
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
.\venv\Scripts\activate.bat

echo Virtual environment activated.
