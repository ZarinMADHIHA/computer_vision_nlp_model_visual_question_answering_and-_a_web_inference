"""
Main training script for VQA fine-tuning
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_strategy import ModelFactory
from dataset_processor import DatasetProcessor
from vqa_manager import VQAManager
from evaluator import VQAEvaluator
import json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train VQA model")

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory for images"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="blip",
        choices=["blip", "clip"],
        help="Model type to use"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-vqa-base",
        help="HuggingFace model name"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/nekoshou/New Volume1/VQA/outputs",
        help="Output directory"
    )

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    print("=" * 70)
    print("VQA Fine-Tuning")
    print("=" * 70)
    print(f"\nModel: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")

    # Set cache directories to external drive
    os.environ['TRANSFORMERS_CACHE'] = '/media/nekoshou/New Volume1/VQA/.cache/transformers'
    os.environ['HF_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/huggingface'
    os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)

    # Step 1: Load dataset
    print("\n" + "=" * 70)
    print("Step 1: Loading Dataset")
    print("=" * 70)

    dataset_processor = DatasetProcessor(args.dataset_path)
    stats = dataset_processor.load_dataset()

    print("\nDataset loaded successfully!")
    print(f"Training samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['val_samples']}")
    print(f"Test samples: {stats['test_samples']}")

    # Step 2: Initialize model
    print("\n" + "=" * 70)
    print("Step 2: Initializing Model")
    print("=" * 70)

    model_strategy = ModelFactory.create_model(args.model_type)
    model_strategy.load_model(args.model_name, args.device)

    print(f"\nModel loaded: {args.model_name}")
    print(f"Model type: {model_strategy.get_model_name()}")

    # Step 3: Create dataloaders
    print("\n" + "=" * 70)
    print("Step 3: Creating DataLoaders")
    print("=" * 70)

    train_loader, val_loader, test_loader = dataset_processor.create_dataloaders(
        processor=model_strategy.processor,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader) if train_loader else 0}")
    print(f"  Val batches: {len(val_loader) if val_loader else 0}")
    print(f"  Test batches: {len(test_loader) if test_loader else 0}")

    # Step 4: Setup training
    print("\n" + "=" * 70)
    print("Step 4: Setting up Training")
    print("=" * 70)

    vqa_manager = VQAManager(
        model_strategy=model_strategy,
        device=args.device,
        output_dir=args.output_dir
    )

    vqa_manager.setup_training(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=0
    )

    # Hyperparameters for logging
    hyperparameters = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_length': args.max_length,
        'device': args.device
    }

    print("\nTraining setup complete!")
    print(f"Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))

    # Step 5: Train model
    print("\n" + "=" * 70)
    print("Step 5: Training Model")
    print("=" * 70)

    history = vqa_manager.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hyperparameters=hyperparameters
    )

    # Step 6: Evaluate on test set
    if test_loader:
        print("\n" + "=" * 70)
        print("Step 6: Evaluating on Test Set")
        print("=" * 70)

        test_metrics = vqa_manager.evaluate(test_loader, "test")
        vqa_manager.evaluator.print_metrics(test_metrics)

        # Save test metrics
        test_metrics_file = Path(args.output_dir) / "test_metrics.json"
        with open(test_metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nTest metrics saved to: {test_metrics_file}")

    # Step 7: Generate sample predictions
    print("\n" + "=" * 70)
    print("Step 7: Generating Sample Predictions")
    print("=" * 70)

    predictions = vqa_manager.predict(
        test_loader if test_loader else val_loader,
        save_results=True
    )

    print(f"\nGenerated {len(predictions)} predictions")
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"\n{i+1}. Image: {pred['image_path']}")
        print(f"   Question: {pred['question']}")
        print(f"   Predicted: {pred['predicted_answer']}")
        print(f"   Ground Truth: {pred['ground_truth']}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
