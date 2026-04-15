@echo off
SETLOCAL

echo ============================================================
echo  Facial Recognition System - Windows Installer
echo ============================================================
echo.

:: Check Python
python --version >NUL 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause & exit /b 1
)

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing build tools (cmake, wheel)...
pip install cmake wheel setuptools

echo.
echo [3/4] Installing dlib (this can take a few minutes)...
pip install dlib

IF ERRORLEVEL 1 (
    echo.
    echo [WARN] dlib build failed. Trying pre-compiled wheel...
    pip install dlib --find-links https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases/download/v19.24.2/
    IF ERRORLEVEL 1 (
        echo [ERROR] Could not install dlib automatically.
        echo         Please install Visual Studio Build Tools from:
        echo         https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo         Then re-run this script.
        pause & exit /b 1
    )
)

echo.
echo [4/4] Installing remaining dependencies...
pip install face_recognition opencv-python Pillow numpy

echo.
echo ============================================================
echo  Installation complete!  Run the app with:
echo     python main.py
echo ============================================================
pause
