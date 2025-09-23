# Root Cause Analysis Runner Script - PowerShell Version
# This script installs dependencies and runs the complete analysis

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "🚀 Root Cause Analysis Runner" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green

try {
    # Get the script directory and change to it
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $ScriptDir
    Write-Host "📁 Working directory: $ScriptDir" -ForegroundColor Cyan

    # Check environment variables
    Write-Host "📋 Checking environment variables..." -ForegroundColor Yellow
    $requiredVars = @(
        'ALIBABA_CLOUD_ACCESS_KEY_ID',
        'ALIBABA_CLOUD_ACCESS_KEY_SECRET',
        'ALIBABA_CLOUD_ROLE_ARN',
        'ALIBABA_CLOUD_ROLE_SESSION_NAME'
    )

    $missingVars = @()
    foreach ($var in $requiredVars) {
        if (-not (Get-ChildItem Env:$var -ErrorAction SilentlyContinue)) {
            $missingVars += $var
        }
    }

    if ($missingVars.Count -gt 0) {
        Write-Host "⚠️ The following environment variables are not set:" -ForegroundColor Yellow
        foreach ($var in $missingVars) {
            Write-Host "   - $var" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "You can set them using:" -ForegroundColor Cyan
        Write-Host "`$env:ALIBABA_CLOUD_ACCESS_KEY_ID = 'your_key_id'" -ForegroundColor Gray
        Write-Host "`$env:ALIBABA_CLOUD_ACCESS_KEY_SECRET = 'your_key_secret'" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Or add them to your system environment variables." -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "✅ All required environment variables are set" -ForegroundColor Green
    }

    # Check if Python is available
    Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Python not found in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3 and ensure it's available as 'python'" -ForegroundColor Red
        throw "Python not available"
    }

    # Set up virtual environment
    Write-Host "🔧 Setting up virtual environment..." -ForegroundColor Yellow
    if (-not (Test-Path "venv")) {
        Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
    }

    # Activate virtual environment
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
    $activateScript = "venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } else {
        throw "Virtual environment activation script not found"
    }

    # Install dependencies
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    Write-Host "   Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }

    Write-Host "   Installing requirements..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependencies"
    }
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green

    # Run the analysis
    Write-Host "🔍 Running root cause analysis on entire dataset..." -ForegroundColor Yellow
    Set-Location notebook
    python root_cause_driver.py all | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed"
    }
    Set-Location ..

    # Check results
    Write-Host "📊 Checking results..." -ForegroundColor Yellow
    $outputFile = "dataset\output.jsonl"
    if (Test-Path $outputFile) {
        Write-Host "✅ Analysis completed successfully" -ForegroundColor Green
        Write-Host "📄 Output file: $outputFile" -ForegroundColor Cyan
        Write-Host ""
        
        # Count results
        $totalProblems = (Get-Content $outputFile).Count
        Write-Host "📈 Total problems processed: $totalProblems" -ForegroundColor Cyan
        Write-Host ""
        
        # Show sample results
        Write-Host "📋 Sample results:" -ForegroundColor Cyan
        Get-Content $outputFile | Select-Object -First 3 | ForEach-Object {
            Write-Host $_ -ForegroundColor Gray
        }
        Write-Host ""
    } else {
        throw "Output file not found: $outputFile"
    }

    Write-Host "🏁 Analysis complete!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Script execution failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
