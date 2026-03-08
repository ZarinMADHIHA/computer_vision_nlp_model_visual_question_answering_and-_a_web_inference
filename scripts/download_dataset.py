"""
Script to download VQA dataset using kagglehub
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import kagglehub


def main():
    """Download VQA dataset"""
    print("=" * 70)
    print("VQA Dataset Download")
    print("=" * 70)

    # Set cache directory to external drive to avoid disk space issues
    external_drive = "/media/nekoshou/New Volume1/VQA"
    cache_dir = os.path.join(external_drive, ".cache", "kagglehub")
    os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables for cache
    os.environ['KAGGLE_DATA_DIR'] = cache_dir

    print(f"\nCache directory: {cache_dir}")
    print("\nDownloading VQA dataset from Kaggle...")
    print("Dataset: bhavikardeshna/visual-question-answering-computer-vision-nlp")

    try:
        # Download latest version
        path = kagglehub.dataset_download(
            "bhavikardeshna/visual-question-answering-computer-vision-nlp"
        )

        print("\n" + "=" * 70)
        print(f"Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        print("=" * 70)

        # List files in dataset
        print("\nDataset contents:")
        dataset_path = Path(path)
        for item in sorted(dataset_path.rglob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(dataset_path)}: {size_mb:.2f} MB")

        # Save path to a file for later use
        path_file = Path(external_drive) / "dataset_path.txt"
        with open(path_file, 'w') as f:
            f.write(path)
        print(f"\nDataset path saved to: {path_file}")

        return path

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. You have Kaggle credentials configured")
        print("2. You have internet connection")
        print("3. You have accepted the dataset terms on Kaggle")
        sys.exit(1)


if __name__ == "__main__":
    main()
