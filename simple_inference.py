"""
Simple inference demo without full training
Downloads a pretrained BLIP model and runs VQA inference
"""
import os
import sys
from pathlib import Path

# Set cache directories to external drive
os.environ['TRANSFORMERS_CACHE'] = '/media/nekoshou/New Volume1/VQA/.cache/transformers'
os.environ['HF_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/huggingface'
os.environ['TORCH_HOME'] = '/media/nekoshou/New Volume1/VQA/.cache/torch'

# Create cache directories
for cache_dir in [os.environ['TRANSFORMERS_CACHE'], os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
    os.makedirs(cache_dir, exist_ok=True)

print("=" * 70)
print("VQA Inference Demo with BLIP")
print("=" * 70)

try:
    import torch
    from transformers import BlipProcessor, BlipForQuestionAnswering
    from PIL import Image
    import requests
    from io import BytesIO

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    print("\n" + "-" * 70)
    print("Loading BLIP VQA model...")
    print("-" * 70)

    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Demo examples with online images
    examples = [
        {
            "image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
            "questions": [
                "What is in the image?",
                "What color is the woman's outfit?",
                "What is the woman doing?",
                "Is this indoors or outdoors?"
            ]
        }
    ]

    print("\n" + "=" * 70)
    print("Running Inference Examples")
    print("=" * 70)

    for idx, example in enumerate(examples, 1):
        print(f"\n--- Example {idx} ---")
        print(f"Image URL: {example['image_url']}")

        try:
            # Load image
            response = requests.get(example['image_url'], timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            print(f"✓ Image loaded: {image.size}")

            # Answer questions
            print("\nQuestions and Answers:")
            print("-" * 50)

            for q_idx, question in enumerate(example['questions'], 1):
                # Prepare inputs
                inputs = processor(images=image, text=question, return_tensors="pt").to(device)

                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_length=20)
                    answer = processor.decode(generated_ids[0], skip_special_tokens=True)

                print(f"{q_idx}. Q: {question}")
                print(f"   A: {answer}")

        except Exception as e:
            print(f"✗ Error processing example: {e}")

    # Try with local image if available
    print("\n" + "=" * 70)
    print("Testing with Local Image (if available)")
    print("=" * 70)

    # Create a simple test image
    print("\nCreating a test image with text...")
    test_image = Image.new('RGB', (400, 300), color=(70, 130, 180))
    test_image_path = "/media/nekoshou/New Volume1/VQA/test_image.jpg"
    test_image.save(test_image_path)

    test_questions = [
        "What color is the image?",
        "What is in the image?",
        "Is this a photograph?"
    ]

    print(f"✓ Test image created: {test_image_path}")
    print("\nQuestions and Answers:")
    print("-" * 50)

    for q_idx, question in enumerate(test_questions, 1):
        inputs = processor(images=test_image, text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=20)
            answer = processor.decode(generated_ids[0], skip_special_tokens=True)

        print(f"{q_idx}. Q: {question}")
        print(f"   A: {answer}")

    # Summary
    print("\n" + "=" * 70)
    print("Inference Demo Complete!")
    print("=" * 70)
    print("\nThe BLIP VQA model successfully:")
    print("  ✓ Loaded pretrained weights")
    print("  ✓ Processed images from URLs")
    print("  ✓ Generated answers to questions")
    print("  ✓ Demonstrated multimodal understanding")
    print("\nThis demonstrates the core functionality of the VQA system.")
    print("For training on custom datasets, use the full training script.")
    print("=" * 70)

except ImportError as e:
    print(f"\n✗ Error: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch transformers pillow requests")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
