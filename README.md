# Visual Question Answering (VQA) Fine-Tuning Project

![Uploading Screencastfrom2026-03-0819-48-57online-video-cutter.com1-ezgif.com-video-to-gif-converter.gif…]()

## Project Overview

This project implements a complete pipeline for fine-tuning multimodal models (BLIP) on Visual Question Answering datasets. The system processes image-text pairs, trains models to answer questions about images, and evaluates performance using comprehensive metrics.

## Features

- **Multimodal Input Processing**: Handles image + text question pairs
- **Model Flexibility**: Strategy pattern implementation for easy model switching (BLIP/CLIP)
- **Comprehensive Evaluation**: Accuracy, F1-score, and BLEU metrics
- **Database Logging**: SQLite database for experiment tracking
- **Visualization**: Automatic generation of training curves and results
- **Checkpointing**: Automatic model checkpointing and recovery
- **GPU Optimization**: Efficient batch processing and caching

## Architecture

### Core Components

#### 1. **DatasetProcessor** (`src/dataset_processor.py`)
- Loads VQA datasets from various formats (JSON, CSV, directory structure)
- Preprocesses image-text pairs
- Creates PyTorch DataLoaders with custom collate functions
- Provides dataset statistics and analysis

#### 2. **VQAModelStrategy** (`src/model_strategy.py`)
- **Strategy Pattern**: Abstract base class for model implementations
- **BLIPStrategy**: Implementation for BLIP VQA models
- **CLIPStrategy**: Implementation for CLIP models
- **ModelFactory**: Factory for creating model instances

#### 3. **VQAManager** (`src/vqa_manager.py`)
- Orchestrates training and evaluation
- Handles optimizer setup and learning rate scheduling
- Implements training loop with gradient accumulation
- Manages checkpointing and model saving
- Generates predictions on test sets

#### 4. **VQAEvaluator** (`src/evaluator.py`)
- Computes accuracy (exact match)
- Calculates F1 scores (weighted and token-level)
- Implements BLEU-N metrics
- Provides error analysis and statistics

#### 5. **VQADatabase** (`src/database.py`)
- SQLite database for experiment logging
- Tables: `VQAExperiments`, `GeneratedAnswers`
- Tracks hyperparameters, losses, and metrics
- Stores generated answers with ground truth

## Database Schema

### VQAExperiments Table
```sql
CREATE TABLE VQAExperiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    hyperparameters TEXT NOT NULL,
    train_loss REAL,
    val_loss REAL,
    metrics TEXT,
    timestamp TEXT NOT NULL
);
```

### GeneratedAnswers Table
```sql
CREATE TABLE GeneratedAnswers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    image_url TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    ground_truth TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES VQAExperiments(id)
);
```

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate vqa_env
```

Or manually:
```bash
conda create -n vqa_env python=3.10
conda activate vqa_env
pip install -r requirements.txt
```

### 2. Configure Cache Directories

To avoid disk space issues, set cache directories to external drive:

```bash
export TRANSFORMERS_CACHE="/media/nekoshou/New Volume1/VQA/.cache/transformers"
export HF_HOME="/media/nekoshou/New Volume1/VQA/.cache/huggingface"
export TORCH_HOME="/media/nekoshou/New Volume1/VQA/.cache/torch"
```

## Usage

### 1. Download Dataset

```bash
python scripts/download_dataset.py
```

This downloads the VQA dataset from Kaggle and saves the path for later use.

### 2. Train Model

Using the training script:

```bash
python scripts/train.py \
    --dataset_path /path/to/dataset \
    --image_root /path/to/images \
    --model_type blip \
    --model_name Salesforce/blip-vqa-base \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir /media/nekoshou/New\ Volume1/VQA/outputs
```

### 3. Using Jupyter Notebook

Open and run `VQA_Fine_Tuning.ipynb` for an interactive experience with visualizations.

```bash
jupyter notebook VQA_Fine_Tuning.ipynb
```

### 4. Visualize Results

```bash
python scripts/visualize_results.py \
    --output_dir /media/nekoshou/New\ Volume1/VQA/outputs \
    --experiment_id 1
```

## Project Structure

```
VQA/
├── src/
│   ├── __init__.py
│   ├── database.py              # Database management
│   ├── dataset_processor.py     # Dataset loading and preprocessing
│   ├── evaluator.py             # Evaluation metrics
│   ├── model_strategy.py        # Model implementations (Strategy pattern)
│   └── vqa_manager.py           # Training orchestration
├── scripts/
│   ├── download_dataset.py      # Dataset download script
│   ├── train.py                 # Training script
│   └── visualize_results.py     # Visualization script
├── outputs/
│   ├── checkpoints/             # Model checkpoints
│   ├── visualizations/          # Generated plots
│   ├── vqa_experiments.db       # Experiment database
│   └── predictions_*.json       # Generated predictions
├── VQA_Fine_Tuning.ipynb        # Jupyter notebook
├── environment.yml              # Conda environment file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Evaluation Metrics

### 1. **Accuracy (Exact Match)**
- Measures the percentage of predictions that exactly match ground truth
- Normalizes text (lowercase, removes articles and punctuation)

### 2. **F1 Score**
- **Weighted F1**: Multi-class F1 with class weights
- **Token F1**: Token-level overlap between prediction and ground truth

### 3. **BLEU Score**
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Measures n-gram overlap between generated and reference answers

## Model Architecture

### BLIP (Bootstrapped Language-Image Pre-training)
- **Vision Encoder**: ViT (Vision Transformer)
- **Text Encoder**: BERT-based encoder
- **Multimodal Fusion**: Cross-attention layers
- **Answer Generation**: Auto-regressive decoder

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5 (default)
- **Weight Decay**: 0.01
- **Gradient Clipping**: Max norm 1.0
- **Batch Size**: 8 (adjustable)
- **Mixed Precision**: Supported via accelerate

## Performance Optimization

1. **Batch Processing**: Efficient DataLoader with custom collate function
2. **GPU Caching**: Model and data caching for faster training
3. **Gradient Accumulation**: Supports larger effective batch sizes
4. **Checkpointing**: Saves best model based on validation loss
5. **Mixed Precision**: Optional FP16 training for faster training

## Design Patterns

### 1. **Strategy Pattern**
- `VQAModelStrategy` abstract base class
- Concrete implementations: `BLIPStrategy`, `CLIPStrategy`
- Easy to add new models without modifying existing code

### 2. **Factory Pattern**
- `ModelFactory` for creating model instances
- Encapsulates model creation logic

### 3. **Manager Pattern**
- `VQAManager` orchestrates training workflow
- Centralizes training logic and state management

## Example Results

After training, you'll find:

1. **Checkpoints**: `outputs/checkpoints/best_model/`
2. **Training History**: JSON file with losses and learning rates
3. **Predictions**: `outputs/predictions_YYYYMMDD_HHMMSS.json`
4. **Metrics**: `outputs/test_metrics.json`
5. **Visualizations**:
   - `training_history.png`
   - `evaluation_metrics.png`
   - `sample_predictions.png`
   - `experiments_table.png`

## Sample Predictions Format

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "question": "What color is the car?",
    "predicted_answer": "red",
    "ground_truth": "red"
  },
  ...
]
```

## Troubleshooting

### Disk Space Issues
- Set cache directories to external drive (see Installation section)
- Clear HuggingFace cache: `rm -rf ~/.cache/huggingface`

### Out of Memory
- Reduce batch size: `--batch_size 4`
- Use gradient accumulation: `--gradient_accumulation_steps 2`
- Reduce image resolution in processor

### Slow Training
- Increase number of workers: `--num_workers 8`
- Use mixed precision training
- Reduce validation frequency

## Citation

If you use this code, please cite the original BLIP paper:

```bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={ICML},
  year={2022}
}
```

## License

This project is for educational and research purposes.

## Contact

For issues and questions, please open an issue on the project repository.

## Acknowledgments

- HuggingFace Transformers library
- Salesforce BLIP model
- OpenAI CLIP model
- Kaggle VQA dataset contributors
