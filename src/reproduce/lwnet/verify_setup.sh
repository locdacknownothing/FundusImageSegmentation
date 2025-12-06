#!/bin/bash
# Quick setup verification script for LWNet reproduction environment

set -e  # Exit on error

echo "==================================="
echo "LWNet Environment Setup Verification"
echo "==================================="
echo ""

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: environment.yml not found. Please run this script from src/reproduce/lwnet/"
    exit 1
fi

echo "✓ Found environment files"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "not found")
if [[ "$python_version" == "3.7"* ]] || [[ "$python_version" == "3.8"* ]] || [[ "$python_version" == "3.9"* ]]; then
    echo "✓ Python $python_version detected (compatible)"
elif [ "$python_version" != "not found" ]; then
    echo "⚠️  Python $python_version detected (LWNet was tested with 3.7, but may work)"
else
    echo "❌ Python 3 not found"
    exit 1
fi
echo ""

# Check if conda is available
echo "Checking for conda..."
if command -v conda &> /dev/null; then
    conda_version=$(conda --version)
    echo "✓ $conda_version found"
    echo ""
    echo "To create the conda environment, run:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate lwnet_reproduce"
else
    echo "⚠️  Conda not found. You can still use pip with requirements.txt"
    echo ""
    echo "To create a virtual environment with pip, run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
fi
echo ""

# Check if lwnet submodule exists
echo "Checking lwnet submodule..."
lwnet_path="../../references/lwnet"
if [ -d "$lwnet_path" ]; then
    echo "✓ LWNet reference directory found at $lwnet_path"
    
    # Check for key files
    if [ -f "$lwnet_path/train_cyclical.py" ]; then
        echo "✓ Training script found"
    else
        echo "❌ Training script not found. Submodule may not be initialized."
        echo "  Run: git submodule update --init --recursive"
    fi
else
    echo "❌ LWNet reference directory not found"
    echo "  Expected at: $lwnet_path"
    echo "  Please initialize the git submodule:"
    echo "  git submodule update --init --recursive"
fi
echo ""

# Summary
echo "==================================="
echo "Verification Summary"
echo "==================================="
echo ""
echo "Environment files:"
echo "  ✓ requirements.txt"
echo "  ✓ environment.yml"
echo "  ✓ README.md"
echo "  ✓ lwnet_colab_training.ipynb"
echo ""
echo "Next steps:"
echo "  1. Choose your setup method (conda or pip)"
echo "  2. Install dependencies"
echo "  3. Navigate to ../../references/lwnet"
echo "  4. Run: python get_public_data.py"
echo "  5. Start training or use the Colab notebook"
echo ""
echo "For detailed instructions, see README.md"
echo "==================================="
