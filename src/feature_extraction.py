"""
feature_extraction.py
----------------------
Uses a pretrained InceptionV3 (ImageNet weights, top layer removed,
global-average pooling) to convert every image in the dataset into a
2048-dimensional feature vector. Each vector is saved individually as
a .npy file so training does not need to reload/re-run the CNN.
"""

import os
import numpy as np
from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
)
from tensorflow.keras.preprocessing.image import img_to_array

from utils import BASE_DIR, FEATURES_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "Flickr8k_Dataset")

TARGET_SIZE = (299, 299)


def build_encoder():
    """Load InceptionV3 with the classification head removed."""
    model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return model


def load_and_preprocess_image(image_path):
    """
    Open an image file, convert to RGB, resize to 299x299, and apply
    InceptionV3's preprocessing. Returns a (1, 299, 299, 3) array or
    None if the file is unreadable/corrupted.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE)
    except (UnidentifiedImageError, OSError):
        return None

    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def extract_features(images_dir, features_dir, model):
    """Extract and save a feature vector for every image in images_dir."""
    os.makedirs(features_dir, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    )
    total = len(image_files)

    if total == 0:
        print(f"[ERROR] No images found in {images_dir}.")
        return

    skipped = []
    for i, filename in enumerate(image_files, start=1):
        image_id = os.path.splitext(filename)[0]
        output_path = os.path.join(features_dir, image_id + ".npy")

        # Skip images that were already processed (resume support)
        if os.path.isfile(output_path):
            if i % 100 == 0 or i == total:
                print(f"Processed {i}/{total}")
            continue

        image_path = os.path.join(images_dir, filename)
        array = load_and_preprocess_image(image_path)

        if array is None:
            skipped.append(filename)
            continue

        feature = model.predict(array, verbose=0)  # shape (1, 2048)
        np.save(output_path, feature[0])

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total}")

    print("Feature extraction completed.")
    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} corrupted/unsupported files:")
        for f in skipped[:10]:
            print(f"   - {f}")
        if len(skipped) > 10:
            print(f"   ...and {len(skipped) - 10} more.")


def main():
    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERROR] Image dataset not found at {IMAGES_DIR}.")
        print("Run check_dataset.py first.")
        return

    print("Loading InceptionV3 (ImageNet weights)...")
    model = build_encoder()

    print("Starting feature extraction...")
    extract_features(IMAGES_DIR, FEATURES_DIR, model)


if __name__ == "__main__":
    main()
