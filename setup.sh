#!/bin/bash

# VQA Project Setup Script
# This script sets up the environment and prepares for training

set -e  # Exit on error

echo "======================================================================"
echo "VQA Fine-Tuning Project Setup"
echo "======================================================================"

# Project directory
PROJECT_DIR="/media/nekoshou/New Volume1/VQA"
cd "$PROJECT_DIR"

# Step 1: Check conda installation
echo ""
echo "Step 1: Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi
echo "✓ Conda found: $(conda --version)"

# Step 2: Check if environment exists
echo ""
echo "Step 2: Checking for vqa_env environment..."
if conda env list | grep -q "vqa_env"; then
    echo "✓ Environment 'vqa_env' already exists"
    read -p "Do you want to reinstall packages? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda activate vqa_env
        pip install -r requirements.txt
        echo "✓ Packages reinstalled"
    fi
else
    echo "Creating new environment 'vqa_env'..."
    conda create -n vqa_env python=3.10 -y
    conda activate vqa_env
    pip install -r requirements.txt
    echo "✓ Environment created and packages installed"
fi

# Step 3: Create cache directories
echo ""
echo "Step 3: Setting up cache directories..."
mkdir -p "$PROJECT_DIR/.cache/transformers"
mkdir -p "$PROJECT_DIR/.cache/huggingface"
mkdir -p "$PROJECT_DIR/.cache/torch"
mkdir -p "$PROJECT_DIR/.cache/kagglehub"
mkdir -p "$PROJECT_DIR/outputs"
mkdir -p "$PROJECT_DIR/outputs/checkpoints"
mkdir -p "$PROJECT_DIR/outputs/visualizations"
echo "✓ Cache directories created"

# Step 4: Set environment variables
echo ""
echo "Step 4: Setting environment variables..."
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/transformers"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export KAGGLE_DATA_DIR="$PROJECT_DIR/.cache/kagglehub"

# Create environment script for future use
cat > "$PROJECT_DIR/set_env.sh" << 'EOL'
#!/bin/bash
# Source this file to set environment variables
export TRANSFORMERS_CACHE="/media/nekoshou/New Volume1/VQA/.cache/transformers"
export HF_HOME="/media/nekoshou/New Volume1/VQA/.cache/huggingface"
export TORCH_HOME="/media/nekoshou/New Volume1/VQA/.cache/torch"
export KAGGLE_DATA_DIR="/media/nekoshou/New Volume1/VQA/.cache/kagglehub"
export PYTHONPATH="/media/nekoshou/New Volume1/VQA/src:$PYTHONPATH"
echo "Environment variables set!"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  HF_HOME=$HF_HOME"
EOL

chmod +x "$PROJECT_DIR/set_env.sh"
echo "✓ Environment variables configured"
echo "  Created set_env.sh for future use"

# Step 5: Check disk space
echo ""
echo "Step 5: Checking disk space..."
AVAIL_SPACE=$(df -BG "$PROJECT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_SPACE" -lt 50 ]; then
    echo "⚠️  Warning: Less than 50GB available. You may need more space for models and datasets."
else
    echo "✓ Sufficient disk space available: ${AVAIL_SPACE}GB"
fi

# Step 6: Check Python packages
echo ""
echo "Step 6: Verifying Python packages..."
conda activate vqa_env
python -c "
import sys
try:
    import torch
    import transformers
    import PIL
    import kagglehub
    print('✓ All required packages installed')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    sys.exit(1)
" || exit 1

# Step 7: Check Kaggle credentials
echo ""
echo "Step 7: Checking Kaggle credentials..."
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ Kaggle credentials found"
else
    echo "⚠️  Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "   To download datasets, you need to:"
    echo "   1. Go to https://www.kaggle.com/settings"
    echo "   2. Create new API token"
    echo "   3. Place kaggle.json in ~/.kaggle/"
    echo "   4. chmod 600 ~/.kaggle/kaggle.json"
fi

# Step 8: Make scripts executable
echo ""
echo "Step 8: Making scripts executable..."
chmod +x scripts/*.py 2>/dev/null || true
echo "✓ Scripts are executable"

# Summary
echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate vqa_env"
echo "  2. Set environment variables: source set_env.sh"
echo "  3. Download dataset: python scripts/download_dataset.py"
echo "  4. Start training: python scripts/train.py --help"
echo ""
echo "Or follow the QUICKSTART.md guide for detailed instructions."
echo ""
echo "Project directory: $PROJECT_DIR"
echo "======================================================================"
