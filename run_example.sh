#!/bin/bash

# Example training run script
# This demonstrates a complete training pipeline

set -e

echo "======================================================================"
echo "VQA Fine-Tuning - Example Training Run"
echo "======================================================================"

# Project setup
PROJECT_DIR="/media/nekoshou/New Volume1/VQA"
cd "$PROJECT_DIR"

# Source environment variables
if [ -f set_env.sh ]; then
    source set_env.sh
else
    echo "Setting environment variables..."
    export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/transformers"
    export HF_HOME="$PROJECT_DIR/.cache/huggingface"
    export TORCH_HOME="$PROJECT_DIR/.cache/torch"
    export KAGGLE_DATA_DIR="$PROJECT_DIR/.cache/kagglehub"
fi

# Activate conda environment
echo ""
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate vqa_env

# Check if dataset path exists
if [ ! -f dataset_path.txt ]; then
    echo ""
    echo "Dataset not found. Downloading..."
    python scripts/download_dataset.py
fi

# Read dataset path
DATASET_PATH=$(cat dataset_path.txt)
echo ""
echo "Using dataset: $DATASET_PATH"

# Training configuration
MODEL_TYPE="blip"
MODEL_NAME="Salesforce/blip-vqa-base"
BATCH_SIZE=8
NUM_EPOCHS=3
LEARNING_RATE=5e-5
OUTPUT_DIR="$PROJECT_DIR/outputs"

echo ""
echo "Training Configuration:"
echo "  Model Type: $MODEL_TYPE"
echo "  Model Name: $MODEL_NAME"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run training
echo ""
echo "Starting training..."
echo "======================================================================"
python scripts/train.py \
    --dataset_path "$DATASET_PATH" \
    --image_root "$DATASET_PATH" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4

echo ""
echo "======================================================================"
echo "Training completed!"
echo "======================================================================"

# Generate visualizations
echo ""
echo "Generating visualizations..."
python scripts/visualize_results.py --output_dir "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "All done!"
echo "======================================================================"
echo ""
echo "Results are available in: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  - Predictions: $OUTPUT_DIR/predictions_*.json"
echo "  - Metrics: $OUTPUT_DIR/test_metrics.json"
echo "  - Visualizations: $OUTPUT_DIR/visualizations/"
echo "  - Database: $OUTPUT_DIR/vqa_experiments.db"
echo ""
echo "To view visualizations:"
echo "  ls $OUTPUT_DIR/visualizations/"
echo ""
echo "To query database:"
echo "  sqlite3 $OUTPUT_DIR/vqa_experiments.db 'SELECT * FROM VQAExperiments;'"
echo ""
