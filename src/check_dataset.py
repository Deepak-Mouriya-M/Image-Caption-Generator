"""
check_dataset.py
-----------------
Verifies that the Flickr8k dataset and extracted features are present
and correctly structured before training or inference begins.

Expected structure:

dataset/
├── Flickr8k_Dataset/       -> contains ~8091 .jpg images
└── Flickr8k_text/
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "Flickr8k_Dataset")
TEXT_DIR = os.path.join(DATASET_DIR, "Flickr8k_text")
FEATURES_DIR = os.path.join(BASE_DIR, "models", "features")

REQUIRED_TEXT_FILES = [
    "Flickr8k.token.txt",
    "Flickr_8k.trainImages.txt",
    "Flickr_8k.devImages.txt",
    "Flickr_8k.testImages.txt",
]

EXPECTED_IMAGE_COUNT = 8091


def check_folder_exists(path, name):
    """Check whether a folder exists and print a status message."""
    if not os.path.isdir(path):
        print(f"[MISSING] {name} folder not found at: {path}")
        return False
    print(f"[OK] {name} folder found at: {path}")
    return True


def count_images(images_dir):
    """Count the number of .jpg / .jpeg / .png images inside a folder."""
    if not os.path.isdir(images_dir):
        return 0
    valid_ext = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    return len(images)


def count_features(features_dir):
    """Count the number of pre-extracted .npy feature files."""
    if not os.path.isdir(features_dir):
        return 0
    return len([f for f in os.listdir(features_dir) if f.endswith(".npy")])


def check_text_files(text_dir):
    """Verify that all required caption/text files are present."""
    missing_files = []
    for filename in REQUIRED_TEXT_FILES:
        file_path = os.path.join(text_dir, filename)
        if not os.path.isfile(file_path):
            missing_files.append(filename)
    return missing_files


def main():
    print("=" * 50)
    print("Dataset Verification - Image Caption Generator")
    print("=" * 50)

    dataset_ok = check_folder_exists(DATASET_DIR, "dataset")
    images_ok = check_folder_exists(IMAGES_DIR, "Flickr8k_Dataset")
    text_ok = check_folder_exists(TEXT_DIR, "Flickr8k_text")

    if not (dataset_ok and images_ok and text_ok):
        print("\nDataset folders are incomplete. Please check dataset placement.")
        return

    image_count = count_images(IMAGES_DIR)
    feature_count = count_features(FEATURES_DIR)
    missing_text_files = check_text_files(TEXT_DIR)

    print(f"\nImages in dataset: {image_count}")
    print(f"Pre-extracted features (.npy): {feature_count}")

    if image_count == 0:
        print("[ERROR] No images found. Please check the dataset folder.")
    elif image_count != EXPECTED_IMAGE_COUNT:
        print(f"[WARNING] Expected {EXPECTED_IMAGE_COUNT} images, found {image_count}.")

    if missing_text_files:
        print("[ERROR] Missing caption files:")
        for f in missing_text_files:
            print(f"   - {f}")
    else:
        print("Caption files found.")

    if image_count > 0 and not missing_text_files:
        print("\nDataset Ready.")
    else:
        print("\nDataset is NOT ready. Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
