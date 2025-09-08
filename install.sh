#!/bin/bash

# PraVaahan AI Scheduler - Installation Script
# This script sets up the development environment for the PraVaahan AI Scheduler

set -e  # Exit on any error

echo "🚂 Setting up PraVaahan AI Scheduler..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.8+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
python -m pytest test_solver.py -v

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To start the service:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "To run tests:"
echo "  source venv/bin/activate"
echo "  python -m pytest test_solver.py -v"
echo ""
echo "To view API documentation:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo "  # Then visit http://localhost:8000/docs"
