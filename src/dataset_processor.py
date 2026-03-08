"""
Dataset processor for VQA
"""
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor
from tqdm import tqdm
import numpy as np


class VQADataset(Dataset):
    """PyTorch Dataset for VQA"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor,
        image_root: str,
        max_length: int = 512
    ):
        """
        Initialize VQA Dataset

        Args:
            data: List of data samples
            processor: BLIP or CLIP processor
            image_root: Root directory for images
            max_length: Maximum sequence length
        """
        self.data = data
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with processed inputs
        """
        sample = self.data[idx]

        # Load image
        image_path = os.path.join(self.image_root, sample['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return dummy image if loading fails
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224))

        # Process image and question
        question = sample['question']
        answer = sample.get('answer', '')

        # Encode inputs
        encoding = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Encode answer as labels
        if answer:
            with self.processor.tokenizer.as_target_tokenizer():
                labels = self.processor.tokenizer(
                    answer,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=20,
                    truncation=True
                )
            encoding['labels'] = labels['input_ids'].squeeze(0)

        # Store metadata
        encoding['question_text'] = question
        encoding['answer_text'] = answer
        encoding['image_path'] = image_path

        return encoding


class DatasetProcessor:
    """Processor for VQA datasets"""

    def __init__(self, dataset_path: str):
        """
        Initialize dataset processor

        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_dataset(self) -> Dict[str, Any]:
        """
        Load VQA dataset from various formats

        Returns:
            Dictionary with dataset statistics
        """
        print(f"Loading dataset from: {self.dataset_path}")

        # Check for JSON format
        json_files = list(self.dataset_path.glob("*.json"))
        if json_files:
            return self._load_json_format(json_files[0])

        # Check for CSV format
        csv_files = list(self.dataset_path.glob("*.csv"))
        if csv_files:
            return self._load_csv_format(csv_files[0])

        # Check for directory structure
        if (self.dataset_path / "train").exists():
            return self._load_directory_format()

        raise ValueError(f"Unknown dataset format in {self.dataset_path}")

    def _load_json_format(self, json_path: Path) -> Dict[str, Any]:
        """Load dataset from JSON file"""
        print(f"Loading JSON format: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Structured JSON with train/val/test splits
            self.train_data = data.get('train', [])
            self.val_data = data.get('val', []) or data.get('validation', [])
            self.test_data = data.get('test', [])
        else:
            # List of samples - need to split
            self.data = data
            self._split_data()

        return self._get_statistics()

    def _load_csv_format(self, csv_path: Path) -> Dict[str, Any]:
        """Load dataset from CSV file"""
        print(f"Loading CSV format: {csv_path}")

        df = pd.read_csv(csv_path)

        # Convert DataFrame to list of dictionaries
        self.data = df.to_dict('records')

        # Check if split column exists
        if 'split' in df.columns:
            self.train_data = df[df['split'] == 'train'].to_dict('records')
            self.val_data = df[df['split'].isin(['val', 'validation'])].to_dict('records')
            self.test_data = df[df['split'] == 'test'].to_dict('records')
        else:
            self._split_data()

        return self._get_statistics()

    def _load_directory_format(self) -> Dict[str, Any]:
        """Load dataset from directory structure"""
        print("Loading directory format")

        # Load train data
        train_path = self.dataset_path / "train"
        self.train_data = self._load_split_from_directory(train_path)

        # Load validation data
        val_path = self.dataset_path / "val"
        if not val_path.exists():
            val_path = self.dataset_path / "validation"
        if val_path.exists():
            self.val_data = self._load_split_from_directory(val_path)
        else:
            self.val_data = []

        # Load test data
        test_path = self.dataset_path / "test"
        if test_path.exists():
            self.test_data = self._load_split_from_directory(test_path)
        else:
            self.test_data = []

        return self._get_statistics()

    def _load_split_from_directory(self, split_path: Path) -> List[Dict[str, Any]]:
        """Load data from a split directory"""
        data = []

        # Check for annotations file
        annotations_file = split_path / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                data = json.load(f)
        else:
            # Fallback: try to infer from file structure
            print(f"Warning: No annotations.json found in {split_path}")

        return data

    def _split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Split data into train/val/test sets

        Args:
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
        """
        if not self.data:
            return

        n = len(self.data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        # Shuffle data
        import random
        random.shuffle(self.data)

        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:train_size + val_size]
        self.test_data = self.data[train_size + val_size:]

    def _get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.train_data) + len(self.val_data) + len(self.test_data),
            'train_samples': len(self.train_data),
            'val_samples': len(self.val_data),
            'test_samples': len(self.test_data)
        }

        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return stats

    def create_dataloaders(
        self,
        processor,
        image_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders

        Args:
            processor: BLIP or CLIP processor
            image_root: Root directory for images
            batch_size: Batch size
            num_workers: Number of worker processes
            max_length: Maximum sequence length

        Returns:
            train_loader, val_loader, test_loader
        """
        # Create datasets
        train_dataset = VQADataset(
            self.train_data, processor, image_root, max_length
        ) if self.train_data else None

        val_dataset = VQADataset(
            self.val_data, processor, image_root, max_length
        ) if self.val_data else None

        test_dataset = VQADataset(
            self.test_data, processor, image_root, max_length
        ) if self.test_data else None

        # Custom collate function
        def collate_fn(batch):
            """Custom collate function"""
            # Separate metadata from tensors
            keys = batch[0].keys()
            tensor_keys = [k for k in keys if isinstance(batch[0][k], torch.Tensor)]
            meta_keys = [k for k in keys if k not in tensor_keys]

            # Collate tensors
            collated = {}
            for key in tensor_keys:
                collated[key] = torch.stack([item[key] for item in batch])

            # Store metadata as lists
            for key in meta_keys:
                collated[key] = [item[key] for item in batch]

            return collated

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        ) if train_dataset else None

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        ) if val_dataset else None

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        ) if test_dataset else None

        return train_loader, val_loader, test_loader

    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze dataset characteristics

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        if self.train_data:
            # Analyze questions
            questions = [item['question'] for item in self.train_data if 'question' in item]
            analysis['avg_question_length'] = np.mean([len(q.split()) for q in questions])

            # Analyze answers
            answers = [item['answer'] for item in self.train_data if 'answer' in item]
            analysis['avg_answer_length'] = np.mean([len(a.split()) for a in answers])

            # Answer distribution
            from collections import Counter
            answer_counts = Counter(answers)
            analysis['unique_answers'] = len(answer_counts)
            analysis['top_10_answers'] = answer_counts.most_common(10)

        return analysis

    def get_train_data(self) -> List[Dict[str, Any]]:
        """Get training data"""
        return self.train_data or []

    def get_val_data(self) -> List[Dict[str, Any]]:
        """Get validation data"""
        return self.val_data or []

    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get test data"""
        return self.test_data or []
