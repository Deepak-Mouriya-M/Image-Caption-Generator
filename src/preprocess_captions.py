"""
preprocess_captions.py
-----------------------
Reads the raw Flickr8k.token.txt file, cleans every caption, wraps
each with startseq/endseq tokens, builds a Keras Tokenizer over the
full vocabulary, and saves the tokenizer + max caption length to disk
for use during training and inference.
"""

import os
import pickle
from collections import defaultdict

from tensorflow.keras.preprocessing.text import Tokenizer

from utils import clean_caption, add_start_end_tokens, BASE_DIR, MODELS_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TEXT_DIR = os.path.join(DATASET_DIR, "Flickr8k_text")
TOKEN_FILE = os.path.join(TEXT_DIR, "Flickr8k.token.txt")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
MAX_LENGTH_PATH = os.path.join(MODELS_DIR, "max_length.pkl")
CLEAN_CAPTIONS_PATH = os.path.join(MODELS_DIR, "captions_clean.pkl")


def load_raw_captions(token_file):
    """
    Parse Flickr8k.token.txt.
    Each line looks like: image_name.jpg#0<TAB>A caption describing it .
    Returns a dict: {image_id: [caption1, caption2, ...]}
    """
    if not os.path.isfile(token_file):
        raise FileNotFoundError(
            f"Caption file not found at {token_file}. "
            "Make sure the dataset is placed correctly (see check_dataset.py)."
        )

    mapping = defaultdict(list)
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on the first tab or whitespace group
            parts = line.split("\t") if "\t" in line else line.split(" ", 1)
            if len(parts) != 2:
                continue
            image_tag, caption = parts
            raw_id = image_tag.split("#")[0]  # strip #0, #1 ... suffix
            image_id = os.path.splitext(raw_id)[0]  # strip .jpg extension for consistent keying
            mapping[image_id].append(caption)
    return mapping


def build_clean_captions(raw_mapping):
    """Apply cleaning + start/end tokens to every caption."""
    clean_mapping = {}
    total_captions = 0
    for image_id, captions in raw_mapping.items():
        cleaned_list = []
        for caption in captions:
            cleaned = clean_caption(caption)
            if len(cleaned.split()) < 2:
                continue  # skip degenerate captions
            wrapped = add_start_end_tokens(cleaned)
            cleaned_list.append(wrapped)
            total_captions += 1
        if cleaned_list:
            clean_mapping[image_id] = cleaned_list
    return clean_mapping, total_captions


def build_tokenizer(clean_mapping):
    """Fit a Keras Tokenizer over every cleaned caption."""
    all_captions = []
    for captions in clean_mapping.values():
        all_captions.extend(captions)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index 0
    max_length = max(len(c.split()) for c in all_captions)
    return tokenizer, vocab_size, max_length


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Reading raw captions...")
    raw_mapping = load_raw_captions(TOKEN_FILE)

    print("Cleaning captions...")
    clean_mapping, total_captions = build_clean_captions(raw_mapping)

    print("Building tokenizer...")
    tokenizer, vocab_size, max_length = build_tokenizer(clean_mapping)

    # Save tokenizer
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save max length
    with open(MAX_LENGTH_PATH, "wb") as f:
        pickle.dump(max_length, f)

    # Save cleaned captions (used later by train_model.py)
    with open(CLEAN_CAPTIONS_PATH, "wb") as f:
        pickle.dump(clean_mapping, f)

    print("\n" + "=" * 40)
    print(f"Images: {len(clean_mapping)}")
    print(f"Captions: {total_captions}")
    print(f"Vocabulary: {vocab_size}")
    print(f"Maximum Length: {max_length}")
    print("=" * 40)
    print(f"\nSaved tokenizer to: {TOKENIZER_PATH}")
    print(f"Saved max_length to: {MAX_LENGTH_PATH}")
    print(f"Saved cleaned captions to: {CLEAN_CAPTIONS_PATH}")


if __name__ == "__main__":
    main()
