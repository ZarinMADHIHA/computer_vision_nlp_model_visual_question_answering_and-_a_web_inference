"""
VQA Manager for training and inference
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from .model_strategy import VQAModelStrategy, ModelFactory
from .evaluator import VQAEvaluator
from .database import VQADatabase


class VQAManager:
    """Manager for VQA model training and inference"""

    def __init__(
        self,
        model_strategy: VQAModelStrategy,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs"
    ):
        """
        Initialize VQA Manager

        Args:
            model_strategy: Model strategy instance
            device: Device to use
            output_dir: Output directory for checkpoints and logs
        """
        self.model_strategy = model_strategy
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = None
        self.scheduler = None
        self.evaluator = VQAEvaluator()
        self.db = VQADatabase(str(self.output_dir / "vqa_experiments.db"))

        self.current_experiment_id = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def setup_training(
        self,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0
    ):
        """
        Setup training components

        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
        """
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model_strategy.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.model_strategy.set_optimizer(self.optimizer)

        # Setup scheduler (optional)
        if warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup
            # Assuming some total steps
            total_steps = warmup_steps * 10
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Average training loss
        """
        self.model_strategy.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
            }

            if 'labels' in batch:
                inputs['labels'] = batch['labels'].to(self.device)

            loss, _ = self.model_strategy.forward(inputs, labels=inputs.get('labels'))

            if loss is not None:
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model_strategy.model.parameters(),
                        max_norm=1.0
                    )

                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

                progress_bar.set_postfix({'loss': total_loss / num_batches})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(
        self,
        val_loader: DataLoader,
        split_name: str = "validation"
    ) -> Dict[str, Any]:
        """
        Evaluate model

        Args:
            val_loader: Validation data loader
            split_name: Name of the split being evaluated

        Returns:
            Dictionary of evaluation metrics
        """
        self.model_strategy.model.eval()
        self.evaluator.reset()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(val_loader, desc=f"Evaluating {split_name}")

        with torch.no_grad():
            for batch in progress_bar:
                # Forward pass
                inputs = {
                    'pixel_values': batch['pixel_values'].to(self.device),
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }

                if 'labels' in batch:
                    inputs['labels'] = batch['labels'].to(self.device)

                loss, _ = self.model_strategy.forward(inputs, labels=inputs.get('labels'))

                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

                # Generate predictions
                batch_size = batch['pixel_values'].size(0)
                for i in range(batch_size):
                    # Get image
                    from PIL import Image
                    image_path = batch['image_path'][i]
                    image = Image.open(image_path).convert('RGB')

                    # Get question
                    question = batch['question_text'][i]

                    # Generate answer
                    predicted_answer = self.model_strategy.generate_answer(image, question)

                    # Get ground truth
                    ground_truth = batch['answer_text'][i]

                    # Add to evaluator
                    self.evaluator.add_batch([predicted_answer], [ground_truth])

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute metrics
        metrics = self.evaluator.compute_all_metrics()
        metrics['loss'] = avg_loss

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 1000,
        eval_steps: int = 500,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Steps between checkpoint saves
            eval_steps: Steps between evaluations
            hyperparameters: Training hyperparameters

        Returns:
            Training history
        """
        # Create experiment record
        if hyperparameters is None:
            hyperparameters = {
                'num_epochs': num_epochs,
                'learning_rate': self.optimizer.defaults['lr'] if self.optimizer else 0,
                'gradient_accumulation_steps': gradient_accumulation_steps
            }

        self.current_experiment_id = self.db.insert_experiment(
            model_name=self.model_strategy.get_model_name(),
            hyperparameters=hyperparameters
        )

        print(f"Starting training - Experiment ID: {self.current_experiment_id}")
        print(f"Model: {self.model_strategy.get_model_name()}")
        print(f"Device: {self.device}")
        print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch(
                train_loader,
                epoch,
                gradient_accumulation_steps
            )
            self.training_history['train_loss'].append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, "validation")
                val_loss = val_metrics['loss']
                self.training_history['val_loss'].append(val_loss)

                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Val F1: {val_metrics['f1_token']:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model")
                    print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}")

            # Update experiment in database
            self.db.update_experiment(
                self.current_experiment_id,
                train_loss=train_loss,
                val_loss=val_metrics['loss'] if val_loader else None,
                metrics=val_metrics if val_loader else None
            )

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

        return self.training_history

    def predict(
        self,
        test_loader: DataLoader,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions on test set

        Args:
            test_loader: Test data loader
            save_results: Whether to save results to database

        Returns:
            List of predictions
        """
        self.model_strategy.model.eval()
        predictions = []

        progress_bar = tqdm(test_loader, desc="Generating predictions")

        with torch.no_grad():
            for batch in progress_bar:
                batch_size = batch['pixel_values'].size(0)

                for i in range(batch_size):
                    # Get image
                    from PIL import Image
                    image_path = batch['image_path'][i]
                    image = Image.open(image_path).convert('RGB')

                    # Get question
                    question = batch['question_text'][i]

                    # Generate answer
                    predicted_answer = self.model_strategy.generate_answer(image, question)

                    # Get ground truth
                    ground_truth = batch['answer_text'][i] if 'answer_text' in batch else ""

                    result = {
                        'image_path': image_path,
                        'question': question,
                        'predicted_answer': predicted_answer,
                        'ground_truth': ground_truth
                    }

                    predictions.append(result)

                    # Save to database
                    if save_results and self.current_experiment_id:
                        self.db.insert_answer(
                            experiment_id=self.current_experiment_id,
                            image_url=image_path,
                            question=question,
                            answer=predicted_answer,
                            ground_truth=ground_truth
                        )

        # Save predictions to file
        if save_results:
            predictions_file = self.output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"\nPredictions saved to: {predictions_file}")

        return predictions

    def save_checkpoint(self, name: str):
        """
        Save model checkpoint

        Args:
            name: Checkpoint name
        """
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model_strategy.save_model(str(checkpoint_dir))

        # Save training history
        history_file = checkpoint_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.model_strategy.load_checkpoint(checkpoint_path)

        # Load training history if available
        history_file = Path(checkpoint_path) / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.training_history = json.load(f)

        print(f"Checkpoint loaded: {checkpoint_path}")
