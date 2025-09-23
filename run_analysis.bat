@echo off
REM Root Cause Analysis Runner Script - Windows Version
REM This script installs dependencies and runs the complete analysis

setlocal enabledelayedexpansion

echo 🚀 Root Cause Analysis Runner
echo ==============================

REM Get the script directory and change to it
cd /d "%~dp0"

REM Load environment variables (Windows doesn't have a direct equivalent to ~/.zshrc)
echo 📋 Loading environment variables...
REM Check if environment variables are set, otherwise provide instructions
if not defined ALIBABA_CLOUD_ACCESS_KEY_ID (
    echo ⚠️ Environment variables not detected. Please ensure the following are set:
    echo    - ALIBABA_CLOUD_ACCESS_KEY_ID
    echo    - ALIBABA_CLOUD_ACCESS_KEY_SECRET  
    echo    - ALIBABA_CLOUD_ROLE_ARN
    echo    - ALIBABA_CLOUD_ROLE_SESSION_NAME
    echo.
    echo You can set them using:
    echo set ALIBABA_CLOUD_ACCESS_KEY_ID=your_key_id
    echo set ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_key_secret
    echo.
    echo Or add them to your system environment variables.
    echo.
)

REM Set up virtual environment
echo 🔧 Setting up virtual environment...
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo ❌ Failed to create virtual environment
        echo Please ensure Python 3 is installed and available as 'python'
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate
if !errorlevel! neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo ❌ Failed to upgrade pip
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Run the analysis
echo 🔍 Running root cause analysis on entire dataset...
cd notebook
python root_cause_driver.py all
if !errorlevel! neq 0 (
    echo ❌ Analysis failed
    cd ..
    pause
    exit /b 1
)
cd ..

REM Check results
echo 📊 Checking results...
if exist "dataset\output.jsonl" (
    echo ✅ Analysis completed successfully
    echo 📄 Output file: dataset\output.jsonl
    echo.
    
    REM Count results (Windows equivalent)
    for /f %%i in ('type "dataset\output.jsonl" ^| find /c /v ""') do set total_problems=%%i
    echo 📈 Total problems processed: !total_problems!
    echo.
    
    REM Show sample results
    echo 📋 Sample results:
    more +1 dataset\output.jsonl | head -3 2>nul || (
        REM Fallback if head is not available
        powershell -command "Get-Content 'dataset\output.jsonl' | Select-Object -First 3"
    )
    echo.
) else (
    echo ❌ Output file not found
    pause
    exit /b 1
)

echo 🏁 Analysis complete!
pause
