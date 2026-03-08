"""
Evaluator module for VQA metrics
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
import re


class VQAEvaluator:
    """Evaluator for VQA models"""

    def __init__(self):
        """Initialize evaluator"""
        self.predictions = []
        self.ground_truths = []

    def add_batch(self, predictions: List[str], ground_truths: List[str]):
        """
        Add a batch of predictions and ground truths

        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text

        Args:
            answer: Answer text

        Returns:
            Normalized answer
        """
        # Convert to lowercase
        answer = answer.lower()

        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)

        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer

    def compute_exact_match(self) -> float:
        """
        Compute exact match accuracy

        Returns:
            Exact match score
        """
        if not self.predictions:
            return 0.0

        normalized_preds = [self.normalize_answer(p) for p in self.predictions]
        normalized_gts = [self.normalize_answer(g) for g in self.ground_truths]

        matches = sum(p == g for p, g in zip(normalized_preds, normalized_gts))
        return matches / len(self.predictions)

    def compute_accuracy(self) -> float:
        """
        Compute accuracy (exact match)

        Returns:
            Accuracy score
        """
        return self.compute_exact_match()

    def compute_f1(self, average: str = 'weighted') -> float:
        """
        Compute F1 score

        Args:
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            F1 score
        """
        if not self.predictions:
            return 0.0

        normalized_preds = [self.normalize_answer(p) for p in self.predictions]
        normalized_gts = [self.normalize_answer(g) for g in self.ground_truths]

        try:
            return f1_score(
                normalized_gts,
                normalized_preds,
                average=average,
                zero_division=0
            )
        except Exception as e:
            print(f"Warning: F1 computation failed: {e}")
            # Fallback to token-level F1
            return self.compute_token_f1()

    def compute_token_f1(self) -> float:
        """
        Compute token-level F1 score

        Returns:
            Average token F1 score
        """
        if not self.predictions:
            return 0.0

        f1_scores = []
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_tokens = set(self.normalize_answer(pred).split())
            gt_tokens = set(self.normalize_answer(gt).split())

            if not gt_tokens:
                continue

            common_tokens = pred_tokens & gt_tokens
            if not common_tokens:
                f1_scores.append(0.0)
            else:
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def compute_bleu(self, n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU score

        Args:
            n: Maximum n-gram order

        Returns:
            Dictionary with BLEU scores
        """
        if not self.predictions:
            return {f'bleu-{i}': 0.0 for i in range(1, n + 1)}

        bleu_scores = {f'bleu-{i}': [] for i in range(1, n + 1)}

        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_tokens = self.normalize_answer(pred).split()
            gt_tokens = self.normalize_answer(gt).split()

            # Compute BLEU for each n-gram order
            for i in range(1, n + 1):
                if len(pred_tokens) < i or len(gt_tokens) < i:
                    bleu_scores[f'bleu-{i}'].append(0.0)
                    continue

                pred_ngrams = self._get_ngrams(pred_tokens, i)
                gt_ngrams = self._get_ngrams(gt_tokens, i)

                matches = sum((pred_ngrams & gt_ngrams).values())
                total = sum(pred_ngrams.values())

                precision = matches / total if total > 0 else 0
                bleu_scores[f'bleu-{i}'].append(precision)

        # Average BLEU scores
        return {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        Get n-grams from tokens

        Args:
            tokens: List of tokens
            n: N-gram order

        Returns:
            Counter of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Accuracy (Exact Match)
        metrics['accuracy'] = self.compute_accuracy()
        metrics['exact_match'] = metrics['accuracy']

        # F1 scores
        metrics['f1_weighted'] = self.compute_f1(average='weighted')
        metrics['f1_token'] = self.compute_token_f1()

        # BLEU scores
        bleu_scores = self.compute_bleu()
        metrics.update(bleu_scores)

        # Additional statistics
        metrics['total_samples'] = len(self.predictions)

        return metrics

    def get_error_analysis(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform error analysis

        Args:
            top_k: Number of top errors to return

        Returns:
            Dictionary with error analysis
        """
        analysis = {}

        # Find incorrect predictions
        errors = []
        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            if self.normalize_answer(pred) != self.normalize_answer(gt):
                errors.append({
                    'index': i,
                    'prediction': pred,
                    'ground_truth': gt
                })

        analysis['num_errors'] = len(errors)
        analysis['error_rate'] = len(errors) / len(self.predictions) if self.predictions else 0
        analysis['sample_errors'] = errors[:top_k]

        # Most common incorrect predictions
        incorrect_preds = [e['prediction'] for e in errors]
        pred_counts = Counter(incorrect_preds)
        analysis['common_errors'] = pred_counts.most_common(top_k)

        return analysis

    def reset(self):
        """Reset evaluator"""
        self.predictions = []
        self.ground_truths = []

    def print_metrics(self, metrics: Dict[str, Any]):
        """
        Print metrics in a formatted way

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 50)
        print("VQA Evaluation Metrics")
        print("=" * 50)

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")

        print("=" * 50 + "\n")
