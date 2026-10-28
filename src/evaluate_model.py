"""
evaluate_model.py
------------------
Evaluates the trained captioning model using BLEU scores (BLEU-1 to BLEU-4),
comparing generated captions against ground-truth reference captions.

Supports evaluating on the official Flickr8k test split (Flickr_8k.testImages.txt)
or on a customizable sample size with configurable beam width.
"""

import os
import argparse
import random
import time

from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model

from utils import (
    BASE_DIR,
    MODELS_DIR,
    load_tokenizer,
    load_max_length,
    load_all_features,
    load_clean_captions,
    MODEL_PATH,
)
from generate_caption import generate_caption

TEST_SPLIT_FILE = os.path.join(BASE_DIR, "dataset", "Flickr8k_text", "Flickr_8k.testImages.txt")


def load_test_ids():
    """Load test image IDs from Flickr_8k.testImages.txt if available."""
    if not os.path.isfile(TEST_SPLIT_FILE):
        return None
    ids = []
    with open(TEST_SPLIT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(os.path.splitext(line)[0])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model with BLEU scores")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of images to evaluate (use 0 for all)")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for caption generation")
    parser.add_argument("--use-test-split", action="store_true", default=True, help="Use official test split if available")
    args = parser.parse_args()

    print("=" * 50)
    print("Evaluating Image Caption Generator Model")
    print("=" * 50)

    print("Loading model, tokenizer, features, and captions...")
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    features = load_all_features()
    clean_captions = load_clean_captions()

    test_split_ids = load_test_ids() if args.use_test_split else None

    if test_split_ids:
        candidate_ids = [k for k in test_split_ids if k in features and k in clean_captions]
        print(f"Using official test split: {len(candidate_ids)} eligible test images.")
    else:
        candidate_ids = [k for k in clean_captions if k in features]
        print(f"Using full dataset pool: {len(candidate_ids)} eligible images.")

    random.seed(42)
    random.shuffle(candidate_ids)

    if args.sample_size and args.sample_size > 0:
        sample_ids = candidate_ids[: args.sample_size]
    else:
        sample_ids = candidate_ids

    total = len(sample_ids)
    print_every = max(1, total // 10)

    references = []
    hypotheses = []

    print(f"\nGenerating captions for {total} images (beam_width={args.beam_width})...")
    start_time = time.time()

    for i, image_id in enumerate(sample_ids, start=1):
        feature = features[image_id].reshape(1, -1)
        res = generate_caption(
            model,
            tokenizer,
            feature,
            max_length,
            beam_width=args.beam_width,
            return_confidence=True,
        )
        if isinstance(res, tuple):
            predicted, confidence = res
        else:
            predicted, confidence = res, 0.0

        predicted_tokens = predicted.lower().rstrip(".").split()

        actual_refs = [
            cap.replace("startseq", "").replace("endseq", "").strip().split()
            for cap in clean_captions[image_id]
        ]

        references.append(actual_refs)
        hypotheses.append(predicted_tokens)

        # Print first 3 samples for inspection
        if i <= 3:
            print(f"\n[Sample {i}] Image: {image_id}")
            print(f"  Predicted:  {predicted} (Confidence: {confidence * 100:.1f}%)")
            print(f"  Reference:  {' '.join(actual_refs[0])}")

        if i % print_every == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"Evaluated {i}/{total} "
                f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)"
            )

    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n" + "=" * 50)
    print("BLEU Score Evaluation Results")
    print("=" * 50)
    print(f"Images Evaluated: {total}")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
