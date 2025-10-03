#!/bin/bash

# Setup script for LOL Bayesian Network Project

echo "=========================================="
echo "LOL Bayesian Network - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the full pipeline:"
echo "  python -m src.cli full --ranks Diamond"
echo ""
echo "For help:"
echo "  python -m src.cli --help"
echo ""


