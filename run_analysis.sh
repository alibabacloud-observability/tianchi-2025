#!/bin/bash

# Root Cause Analysis Runner Script
# This script installs dependencies and runs the complete analysis

set -e  # Exit on any error

echo "🚀 Root Cause Analysis Runner"
echo "=============================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
echo "📋 Loading environment variables..."
# Only source if running in zsh, otherwise skip
if [ -n "$ZSH_VERSION" ]; then
    source ~/.zshrc 2>/dev/null || echo "⚠️ Could not load ~/.zshrc, continuing..."
else
    echo "⚠️ Not running in zsh, skipping ~/.zshrc"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the analysis
echo "🔍 Running root cause analysis on entire dataset..."
cd notebook
python3 root_cause_driver.py all

# Check results
echo "📊 Checking results..."
if [ -f "../dataset/output.jsonl" ]; then
    echo "✅ Analysis completed successfully"
    echo "📄 Output file: dataset/output.jsonl"

    # Count results
    total_problems=$(wc -l < ../dataset/output.jsonl)
    echo "📈 Total problems processed: $total_problems"

    # Show sample results
    echo "📋 Sample results:"
    head -3 ../dataset/output.jsonl

else
    echo "❌ Output file not found"
    exit 1
fi

echo "🏁 Analysis complete!"
