"""
Strategy pattern for VQA model implementations
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    CLIPProcessor, CLIPModel
)
from PIL import Image


class VQAModelStrategy(ABC):
    """Abstract base class for VQA model strategies"""

    @abstractmethod
    def load_model(self, model_name: str, device: str):
        """Load the model and processor"""
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Preprocess image and question"""
        pass

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through the model"""
        pass

    @abstractmethod
    def generate_answer(self, image: Image.Image, question: str) -> str:
        """Generate answer for a question about an image"""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Perform a single training step"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass


class BLIPStrategy(VQAModelStrategy):
    """BLIP model strategy for VQA"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.optimizer = None

    def load_model(self, model_name: str = "Salesforce/blip-vqa-base", device: str = "cuda"):
        """
        Load BLIP model and processor

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
        """
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model_name = model_name

    def preprocess(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """
        Preprocess image and question for BLIP

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Preprocessed inputs
        """
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs: Dict[str, Any], labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BLIP

        Args:
            inputs: Preprocessed inputs
            labels: Target labels (optional)

        Returns:
            loss, logits
        """
        if labels is not None:
            outputs = self.model(**inputs, labels=labels)
            return outputs.loss, outputs.logits
        else:
            outputs = self.model(**inputs)
            return None, outputs.logits

    def generate_answer(self, image: Image.Image, question: str, max_length: int = 20) -> str:
        """
        Generate answer using BLIP

        Args:
            image: PIL Image
            question: Question text
            max_length: Maximum answer length

        Returns:
            Generated answer text
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(**inputs, max_length=max_length)
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return answer

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform a single training step

        Args:
            batch: Batch of training data

        Returns:
            Loss value
        """
        self.model.train()

        # Move batch to device
        inputs = {
            'pixel_values': batch['pixel_values'].to(self.device),
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

        if 'labels' in batch:
            inputs['labels'] = batch['labels'].to(self.device)

        outputs = self.model(**inputs)
        loss = outputs.loss

        return loss.item()

    def get_model_name(self) -> str:
        """Get the model name"""
        return "BLIP"

    def set_optimizer(self, optimizer):
        """Set the optimizer"""
        self.optimizer = optimizer

    def save_model(self, save_path: str):
        """Save model checkpoint"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.model = BlipForQuestionAnswering.from_pretrained(checkpoint_path)
        self.processor = BlipProcessor.from_pretrained(checkpoint_path)
        self.model.to(self.device)


class CLIPStrategy(VQAModelStrategy):
    """CLIP model strategy for VQA"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.optimizer = None

    def load_model(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        Load CLIP model and processor

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
        """
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model_name = model_name

        # Add custom answer head for VQA
        # Note: CLIP is primarily for image-text similarity, not generative VQA
        # This is a simplified implementation

    def preprocess(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """
        Preprocess image and question for CLIP

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Preprocessed inputs
        """
        inputs = self.processor(
            text=[question],
            images=image,
            return_tensors="pt",
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CLIP

        Args:
            inputs: Preprocessed inputs

        Returns:
            loss, logits
        """
        outputs = self.model(**inputs)
        # For CLIP, we use image-text similarity
        logits = outputs.logits_per_image
        return None, logits

    def generate_answer(self, image: Image.Image, question: str, answer_candidates: List[str] = None) -> str:
        """
        Generate answer using CLIP (ranking approach)

        Note: CLIP is better suited for classification among candidates
        rather than generative VQA

        Args:
            image: PIL Image
            question: Question text
            answer_candidates: List of possible answers

        Returns:
            Selected answer
        """
        if answer_candidates is None:
            # Default candidates for yes/no questions
            answer_candidates = ["yes", "no", "maybe", "unknown"]

        self.model.eval()
        with torch.no_grad():
            # Combine question with each answer candidate
            text_inputs = [f"{question} {ans}" for ans in answer_candidates]

            inputs = self.processor(
                text=text_inputs,
                images=[image] * len(answer_candidates),
                return_tensors="pt",
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            best_idx = probs.argmax().item()

        return answer_candidates[best_idx]

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform a single training step

        Note: Training CLIP for VQA requires custom loss

        Args:
            batch: Batch of training data

        Returns:
            Loss value
        """
        self.model.train()

        inputs = {
            'pixel_values': batch['pixel_values'].to(self.device),
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

        outputs = self.model(**inputs)

        # Custom loss computation for VQA
        # This is a simplified version
        logits = outputs.logits_per_image
        labels = torch.arange(len(logits)).to(self.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss.item()

    def get_model_name(self) -> str:
        """Get the model name"""
        return "CLIP"

    def set_optimizer(self, optimizer):
        """Set the optimizer"""
        self.optimizer = optimizer

    def save_model(self, save_path: str):
        """Save model checkpoint"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.model = CLIPModel.from_pretrained(checkpoint_path)
        self.processor = CLIPProcessor.from_pretrained(checkpoint_path)
        self.model.to(self.device)


class ModelFactory:
    """Factory for creating model strategies"""

    @staticmethod
    def create_model(model_type: str = "blip") -> VQAModelStrategy:
        """
        Create a model strategy

        Args:
            model_type: Type of model ("blip" or "clip")

        Returns:
            VQAModelStrategy instance
        """
        if model_type.lower() == "blip":
            return BLIPStrategy()
        elif model_type.lower() == "clip":
            return CLIPStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
