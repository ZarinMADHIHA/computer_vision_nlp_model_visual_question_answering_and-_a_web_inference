"""
Script to visualize VQA training results and create evaluation plots
"""
import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database import VQADatabase


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize VQA results")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/nekoshou/New Volume1/VQA/outputs",
        help="Output directory with results"
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=None,
        help="Experiment ID to visualize (default: latest)"
    )

    return parser.parse_args()


def plot_training_history(history: dict, output_dir: Path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot learning rate if available
    ax = axes[1]
    if history.get('learning_rate'):
        ax.plot(history['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data available',
                ha='center', va='center', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    save_path = output_dir / 'training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot: {save_path}")
    plt.close()


def plot_metrics(metrics: dict, output_dir: Path):
    """Plot evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Overall metrics
    ax = axes[0, 0]
    metric_names = ['Accuracy', 'F1 (Token)', 'F1 (Weighted)']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('f1_token', 0),
        metrics.get('f1_weighted', 0)
    ]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    # BLEU scores
    ax = axes[0, 1]
    bleu_scores = [
        metrics.get('bleu-1', 0),
        metrics.get('bleu-2', 0),
        metrics.get('bleu-3', 0),
        metrics.get('bleu-4', 0)
    ]
    x = np.arange(1, 5)
    ax.plot(x, bleu_scores, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('N-gram Order', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('BLEU Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Metrics comparison (radar chart would be nice, but bar for simplicity)
    ax = axes[1, 0]
    all_metrics = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != 'total_samples' and k != 'loss'}
    if all_metrics:
        metric_names = list(all_metrics.keys())[:6]  # Top 6 metrics
        metric_values = [all_metrics[k] for k in metric_names]
        ax.barh(metric_names, metric_values, color='#1abc9c', alpha=0.7)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Detailed Metrics', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Evaluation Summary
    {'='*40}

    Total Samples: {metrics.get('total_samples', 0)}

    Accuracy: {metrics.get('accuracy', 0):.4f}
    F1 Score (Token): {metrics.get('f1_token', 0):.4f}
    F1 Score (Weighted): {metrics.get('f1_weighted', 0):.4f}

    BLEU-1: {metrics.get('bleu-1', 0):.4f}
    BLEU-2: {metrics.get('bleu-2', 0):.4f}
    BLEU-3: {metrics.get('bleu-3', 0):.4f}
    BLEU-4: {metrics.get('bleu-4', 0):.4f}

    Loss: {metrics.get('loss', 0):.4f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    save_path = output_dir / 'evaluation_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot: {save_path}")
    plt.close()


def create_metrics_table(db: VQADatabase, output_dir: Path):
    """Create a table of all experiments"""
    experiments = db.get_all_experiments()

    if not experiments:
        print("No experiments found")
        return

    # Convert to DataFrame
    rows = []
    for exp in experiments:
        row = {
            'Experiment ID': exp['id'],
            'Model': exp['model_name'],
            'Train Loss': f"{exp['train_loss']:.4f}" if exp['train_loss'] else 'N/A',
            'Val Loss': f"{exp['val_loss']:.4f}" if exp['val_loss'] else 'N/A',
            'Timestamp': exp['timestamp']
        }

        # Add metrics if available
        if exp['metrics']:
            metrics = exp['metrics']
            row['Accuracy'] = f"{metrics.get('accuracy', 0):.4f}"
            row['F1'] = f"{metrics.get('f1_token', 0):.4f}"
            row['BLEU-4'] = f"{metrics.get('bleu-4', 0):.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = output_dir / 'experiments_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved experiments summary: {csv_path}")

    # Create pretty plot
    fig, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12] * len(df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('VQA Experiments Summary', fontsize=16, fontweight='bold', pad=20)
    save_path = output_dir / 'experiments_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved experiments table: {save_path}")
    plt.close()


def visualize_sample_predictions(predictions_file: Path, output_dir: Path, num_samples: int = 10):
    """Visualize sample predictions"""
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Sample predictions
    import random
    samples = random.sample(predictions, min(num_samples, len(predictions)))

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, sample in enumerate(samples[:10]):
        ax = axes[i]

        # Try to load and display image
        try:
            from PIL import Image
            img = Image.open(sample['image_path'])
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Image not\navailable',
                    ha='center', va='center', fontsize=10)

        ax.axis('off')

        # Add question and answers
        question = sample['question'][:50] + '...' if len(sample['question']) > 50 else sample['question']
        pred = sample['predicted_answer'][:20] + '...' if len(sample['predicted_answer']) > 20 else sample['predicted_answer']
        gt = sample['ground_truth'][:20] + '...' if len(sample['ground_truth']) > 20 else sample['ground_truth']

        title = f"Q: {question}\nPred: {pred}\nGT: {gt}"
        ax.set_title(title, fontsize=8, wrap=True)

        # Color border based on correctness
        if sample['predicted_answer'].lower().strip() == sample['ground_truth'].lower().strip():
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

    plt.tight_layout()
    save_path = output_dir / 'sample_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample predictions: {save_path}")
    plt.close()


def main():
    """Main visualization function"""
    args = parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("VQA Results Visualization")
    print("=" * 70)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Initialize database
    db_path = output_dir / "vqa_experiments.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    db = VQADatabase(str(db_path))

    # Get experiment
    if args.experiment_id:
        experiment = db.get_experiment(args.experiment_id)
        if not experiment:
            print(f"Experiment {args.experiment_id} not found")
            return
    else:
        # Get latest experiment
        experiments = db.get_all_experiments()
        if not experiments:
            print("No experiments found")
            return
        experiment = experiments[0]

    print(f"\nVisualizing Experiment ID: {experiment['id']}")
    print(f"Model: {experiment['model_name']}")
    print(f"Timestamp: {experiment['timestamp']}")

    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.exists():
        # Find latest checkpoint with history
        for checkpoint_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
            history_file = checkpoint_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                plot_training_history(history, viz_dir)
                break

    # Plot metrics
    if experiment['metrics']:
        plot_metrics(experiment['metrics'], viz_dir)

    # Create experiments table
    create_metrics_table(db, viz_dir)

    # Visualize sample predictions
    predictions_files = list(output_dir.glob("predictions_*.json"))
    if predictions_files:
        latest_predictions = sorted(predictions_files)[-1]
        visualize_sample_predictions(latest_predictions, viz_dir)

    print("\n" + "=" * 70)
    print("Visualization completed!")
    print(f"Results saved to: {viz_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
