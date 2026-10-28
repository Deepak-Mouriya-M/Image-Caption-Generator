"""
utils.py
--------
Reusable helper functions used across preprocessing, training, and
caption generation scripts. Keeping these in one place avoids
duplicated logic and makes the rest of the code easier to read.
"""

import os
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(MODELS_DIR, "features")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
MAX_LENGTH_PATH = os.path.join(MODELS_DIR, "max_length.pkl")
CLEAN_CAPTIONS_PATH = os.path.join(MODELS_DIR, "captions_clean.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "caption_model.keras")


# ---------------------------------------------------------------------
# Loading saved artifacts
# ---------------------------------------------------------------------

def load_tokenizer():
    """Load the Keras tokenizer saved during preprocessing."""
    if not os.path.isfile(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run preprocess_captions.py first."
        )
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def load_max_length():
    """Load the maximum caption length saved during preprocessing."""
    if not os.path.isfile(MAX_LENGTH_PATH):
        raise FileNotFoundError(
            f"max_length.pkl not found at {MAX_LENGTH_PATH}. "
            "Run preprocess_captions.py first."
        )
    with open(MAX_LENGTH_PATH, "rb") as f:
        max_length = pickle.load(f)
    return max_length


def load_clean_captions(path=CLEAN_CAPTIONS_PATH):
    """
    Load cleaned captions mapping and ensure keys are normalized
    (no file extension, e.g. '1000268201_693b08cb0e').
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Clean captions file not found at {path}. "
            "Run preprocess_captions.py first."
        )
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {os.path.splitext(k)[0]: v for k, v in raw.items()}


def load_feature(image_name):
    """
    Load a single pre-extracted 2048-dim feature vector for an image.
    image_name should be the base filename, e.g. '1000268201.jpg'
    """
    image_id = os.path.splitext(image_name)[0]
    feature_path = os.path.join(FEATURES_DIR, image_id + ".npy")
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(
            f"Feature file not found for {image_name} at {feature_path}. "
            "Run feature_extraction.py first."
        )
    return np.load(feature_path)


def load_all_features():
    """Load every .npy feature file into a dict keyed by image_id."""
    if not os.path.isdir(FEATURES_DIR) or len(os.listdir(FEATURES_DIR)) == 0:
        raise FileNotFoundError(
            f"No feature files found in {FEATURES_DIR}. "
            "Run feature_extraction.py first."
        )
    features = {}
    for filename in os.listdir(FEATURES_DIR):
        if filename.endswith(".npy"):
            image_id = filename.replace(".npy", "")
            features[image_id] = np.load(os.path.join(FEATURES_DIR, filename))
    return features


# ---------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------

def clean_caption(caption):
    """
    Clean a raw caption string:
    - lowercase
    - remove punctuation and numbers
    - remove extra whitespace
    - remove single-character 'words' (e.g. leftover 's', 'a' is kept
      since it is a valid English word, but stray letters are dropped)
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    words = [w for w in caption.split() if len(w) > 1 or w == "a"]
    return " ".join(words)


def add_start_end_tokens(caption):
    """Wrap a cleaned caption with startseq / endseq markers."""
    return f"startseq {caption} endseq"


# ---------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------

def pad_caption_sequence(sequence, max_length, padding="pre"):
    """Pad a tokenized sequence to max_length (defaults to pre-padding for LSTM)."""
    return pad_sequences([sequence], maxlen=max_length, padding=padding)[0]


def word_for_id(integer, tokenizer):
    """Convert a predicted integer index back into its word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def index_to_word(tokenizer):
    """Build a reverse lookup dict {index: word} for faster decoding."""
    return {index: word for word, index in tokenizer.word_index.items()}
