# VQA Fine-Tuning Quick Start Guide

## Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 120GB+ free disk space on external drive
- Kaggle account (for dataset download)

## Step 1: Environment Setup

```bash
# Navigate to project directory
cd "/media/nekoshou/New Volume1/VQA"

# Activate the conda environment
conda activate vqa_env

# If environment doesn't exist or has missing packages, install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Cache Directories

```bash
# Set environment variables to use external drive (avoids disk space issues)
export TRANSFORMERS_CACHE="/media/nekoshou/New Volume1/VQA/.cache/transformers"
export HF_HOME="/media/nekoshou/New Volume1/VQA/.cache/huggingface"
export TORCH_HOME="/media/nekoshou/New Volume1/VQA/.cache/torch"
export KAGGLE_DATA_DIR="/media/nekoshou/New Volume1/VQA/.cache/kagglehub"

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $KAGGLE_DATA_DIR
```

## Step 3: Download Dataset

```bash
# Download VQA dataset from Kaggle
python scripts/download_dataset.py
```

This will download the dataset and save the path to `dataset_path.txt`.

## Step 4: Quick Training Test

### Option A: Using Python Script (Recommended)

```bash
# Read dataset path
DATASET_PATH=$(cat dataset_path.txt)

# Run training with minimal config for testing
python scripts/train.py \
    --dataset_path "$DATASET_PATH" \
    --image_root "$DATASET_PATH" \
    --model_type blip \
    --model_name Salesforce/blip-vqa-base \
    --batch_size 4 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --output_dir "/media/nekoshou/New Volume1/VQA/outputs"
```

### Option B: Using Jupyter Notebook

```bash
# Start Jupyter notebook
jupyter notebook VQA_Fine_Tuning.ipynb
```

Then run all cells in the notebook.

## Step 5: Full Training

For complete training with better results:

```bash
python scripts/train.py \
    --dataset_path "$DATASET_PATH" \
    --image_root "$DATASET_PATH" \
    --model_type blip \
    --model_name Salesforce/blip-vqa-base \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --output_dir "/media/nekoshou/New Volume1/VQA/outputs"
```

## Step 6: Visualize Results

```bash
python scripts/visualize_results.py \
    --output_dir "/media/nekoshou/New Volume1/VQA/outputs"
```

Visualizations will be saved in `outputs/visualizations/`.

## Expected Output Files

After training, you should find:

```
outputs/
├── checkpoints/
│   ├── best_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── training_history.json
│   ├── epoch_1/
│   ├── epoch_2/
│   └── epoch_3/
├── visualizations/
│   ├── training_history.png
│   ├── evaluation_metrics.png
│   ├── sample_predictions.png
│   └── experiments_table.png
├── vqa_experiments.db
├── predictions_YYYYMMDD_HHMMSS.json
└── test_metrics.json
```

## Troubleshooting

### Issue: Out of Disk Space

**Solution**: Ensure cache directories point to external drive:
```bash
# Check current cache locations
echo $TRANSFORMERS_CACHE
echo $HF_HOME

# Clear old cache if needed
rm -rf ~/.cache/huggingface/*
rm -rf ~/.cache/torch/*
```

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size and use gradient accumulation:
```bash
python scripts/train.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    ...
```

### Issue: Dataset Not Found

**Solution**:
1. Check if `dataset_path.txt` exists and contains valid path
2. Verify Kaggle credentials: `~/.kaggle/kaggle.json`
3. Re-run download script

### Issue: Slow Training

**Solution**:
1. Increase num_workers: `--num_workers 8`
2. Use smaller validation set for faster epochs
3. Reduce max_length: `--max_length 256`

## Performance Tips

1. **Use GPU**: Training on GPU is ~10-50x faster than CPU
2. **Batch Size**: Larger batch size = faster training (if GPU memory allows)
3. **Gradient Accumulation**: Simulates larger batch size without memory increase
4. **Mixed Precision**: Add `--fp16` flag (requires apex or accelerate)

## Validation

To verify everything works:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check HuggingFace
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# List available disk space
df -h "/media/nekoshou/New Volume1"
```

## Next Steps

1. **Experiment with hyperparameters**:
   - Try different learning rates: `--learning_rate 1e-5` or `--learning_rate 1e-4`
   - Adjust batch size: `--batch_size 16` (if GPU allows)
   - Train longer: `--num_epochs 5`

2. **Try different models**:
   - Larger BLIP: `--model_name Salesforce/blip-vqa-capfilt-large`
   - CLIP: `--model_type clip --model_name openai/clip-vit-base-patch32`

3. **Analyze results**:
   - Review metrics in `test_metrics.json`
   - Examine predictions in `predictions_*.json`
   - Check visualizations in `outputs/visualizations/`

4. **Human evaluation**:
   - Sample predictions and manually verify answer correctness
   - Identify common error patterns
   - Refine training based on findings

## Quick Commands Summary

```bash
# Setup
cd "/media/nekoshou/New Volume1/VQA"
conda activate vqa_env

# Download dataset
python scripts/download_dataset.py

# Train model
python scripts/train.py --dataset_path $(cat dataset_path.txt) --image_root $(cat dataset_path.txt) --batch_size 8 --num_epochs 3

# Visualize results
python scripts/visualize_results.py

# View results
ls -lh outputs/
```

## Getting Help

- Check README.md for detailed documentation
- Review error messages carefully
- Ensure all prerequisites are met
- Verify disk space availability

## Success Indicators

Training is successful if you see:
- ✅ Training loss decreasing over epochs
- ✅ Validation accuracy > 40%
- ✅ Generated predictions look reasonable
- ✅ Checkpoints saved without errors
- ✅ Visualizations generated successfully

Enjoy fine-tuning your VQA model!
