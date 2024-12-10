@echo off
REM Install a Python package and add it to requirements.txt if successful.

REM Check if package name is provided
if "%~1"=="" (
    echo Usage: install_package.bat [package-name]
    exit /b 1
)

set PACKAGE_NAME=%~1

REM Install the package
echo Installing %PACKAGE_NAME%...
pip install %PACKAGE_NAME%

REM Check if installation was successful
pip freeze | findstr /i "^%PACKAGE_NAME%==" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install %PACKAGE_NAME%.
    exit /b 1
)

REM Get the package and version
for /f "delims=" %%i in ('pip freeze ^| findstr /i "^%PACKAGE_NAME%=="') do set PACKAGE_INFO=%%i

REM Append to requirements.txt
echo %PACKAGE_INFO% >> requirements.txt
echo %PACKAGE_INFO% added to requirements.txt

exit /b 0
