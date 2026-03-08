"""
Comprehensive VQA Demo using our custom classes
Demonstrates the Strategy pattern and evaluation metrics
"""
import os
import sys
from pathlib import Path

# Set cache directories
os.environ['TRANSFORMERS_CACHE'] = '/media/nekoshou/New Volume1/VQA/.cache/transformers'
os.environ['HF_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/huggingface'
os.environ['TORCH_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/torch'

# Create cache directories
for cache_dir in [os.environ['TRANSFORMERS_CACHE'], os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
    os.makedirs(cache_dir, exist_ok=True)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("=" * 70)
print("VQA System Demo with Custom Implementation")
print("=" * 70)

try:
    import torch
    from PIL import Image
    import requests
    from io import BytesIO

    from model_strategy import ModelFactory, BLIPStrategy
    from evaluator import VQAEvaluator
    from database import VQADatabase

    print("\n✓ All modules imported successfully")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    # Demonstrate Strategy Pattern
    print("\n" + "=" * 70)
    print("1. Strategy Pattern Demonstration")
    print("=" * 70)

    print("\nCreating BLIP strategy using ModelFactory...")
    model_strategy = ModelFactory.create_model("blip")
    print(f"✓ Strategy created: {model_strategy.__class__.__name__}")

    print("\nLoading BLIP model...")
    model_strategy.load_model("Salesforce/blip-vqa-base", device)
    print(f"✓ Model loaded: {model_strategy.get_model_name()}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model_strategy.model.parameters()):,}")

    # Demonstrate Inference
    print("\n" + "=" * 70)
    print("2. VQA Inference Demonstration")
    print("=" * 70)

    # Test examples
    test_data = [
        {
            "image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
            "questions": [
                ("What is in the image?", "woman and dog"),
                ("What color is the woman's outfit?", "black and white"),
                ("What is the woman doing?", "petting dog"),
                ("Is this indoors or outdoors?", "outdoors")
            ]
        }
    ]

    predictions = []
    ground_truths = []

    for example in test_data:
        print(f"\nImage: {example['image_url']}")

        # Load image
        response = requests.get(example['image_url'], timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✓ Image loaded: {image.size}")

        print("\nQuestions and Answers:")
        print("-" * 70)

        for question, ground_truth in example['questions']:
            # Use our strategy to generate answer
            answer = model_strategy.generate_answer(image, question)

            predictions.append(answer)
            ground_truths.append(ground_truth)

            # Check if correct
            match = "✓" if answer.lower().strip() in ground_truth.lower() else "✗"

            print(f"Q: {question}")
            print(f"   Predicted: {answer}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   {match}")
            print()

    # Demonstrate Evaluator
    print("=" * 70)
    print("3. Evaluation Metrics Demonstration")
    print("=" * 70)

    evaluator = VQAEvaluator()
    evaluator.add_batch(predictions, ground_truths)

    # Compute all metrics
    metrics = evaluator.compute_all_metrics()

    print("\nEvaluation Results:")
    evaluator.print_metrics(metrics)

    # Error analysis
    error_analysis = evaluator.get_error_analysis(top_k=5)
    print("Error Analysis:")
    print(f"  Total errors: {error_analysis['num_errors']}")
    print(f"  Error rate: {error_analysis['error_rate']:.2%}")

    # Demonstrate Database Logging
    print("=" * 70)
    print("4. Database Logging Demonstration")
    print("=" * 70)

    db_path = "/media/nekoshou/New Volume1/VQA/outputs/demo_vqa.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with VQADatabase(db_path) as db:
        # Create experiment
        hyperparameters = {
            "model_name": "Salesforce/blip-vqa-base",
            "model_type": "blip",
            "device": device
        }

        experiment_id = db.insert_experiment(
            model_name="BLIP",
            hyperparameters=hyperparameters,
            metrics=metrics
        )

        print(f"\n✓ Experiment created with ID: {experiment_id}")

        # Log predictions
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            db.insert_answer(
                experiment_id=experiment_id,
                image_url=test_data[0]['image_url'],
                question=test_data[0]['questions'][i % len(test_data[0]['questions'])][0],
                answer=pred,
                ground_truth=gt
            )

        print(f"✓ Logged {len(predictions)} predictions to database")

        # Retrieve experiment
        retrieved_exp = db.get_experiment(experiment_id)
        print(f"\n✓ Retrieved experiment from database:")
        print(f"   Model: {retrieved_exp['model_name']}")
        print(f"   Timestamp: {retrieved_exp['timestamp']}")
        print(f"   Accuracy: {retrieved_exp['metrics']['accuracy']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    print("\n✅ Successfully Demonstrated:")
    print("  1. Strategy Pattern (ModelFactory + BLIPStrategy)")
    print("  2. VQA Inference with BLIP model")
    print("  3. Comprehensive Evaluation Metrics")
    print("     - Accuracy (Exact Match)")
    print("     - F1 Score (Weighted & Token-level)")
    print("     - BLEU Scores (1-4)")
    print("  4. SQLite Database Logging")
    print("     - VQAExperiments table")
    print("     - GeneratedAnswers table")

    print("\n📊 Results Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1_token']:.4f}")
    print(f"  BLEU-4: {metrics['bleu-4']:.4f}")
    print(f"  Total Samples: {metrics['total_samples']}")

    print("\n💾 Database Location:")
    print(f"  {db_path}")

    print("\n" + "=" * 70)
    print("This demonstrates the complete VQA pipeline:")
    print("  - Modular OOP design")
    print("  - Strategy pattern for model switching")
    print("  - Comprehensive evaluation")
    print("  - Persistent experiment tracking")
    print("=" * 70)

except ImportError as e:
    print(f"\n✗ Error: Missing required package: {e}")
    print("\nPlease ensure all packages are installed.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error during demo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
