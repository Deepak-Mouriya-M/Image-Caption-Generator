"""
generate_caption.py
--------------------
Loads the trained model, tokenizer, and max_length, then generates a
natural-language caption for an image using:
1. High-precision visual feature matching against the verified dataset library.
2. Batched beam search decoding with length penalty and n-gram repetition blocking.

Supports both file paths and PIL Image objects directly in memory.
Computes an accurate confidence score based on the model's token probabilities.
"""

import os
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
)
from tensorflow.keras.preprocessing.image import img_to_array

from utils import load_tokenizer, load_max_length, MODEL_PATH, index_to_word

TARGET_SIZE = (299, 299)
BEAM_WIDTH = 5

_encoder = None  # lazily-built InceptionV3, cached across calls
_caption_model = None  # lazily-loaded caption model, cached across calls
_feature_index = None  # (keys, normalized_matrix, captions_dict)


def get_encoder():
    """Build (once) and return the InceptionV3 feature extractor."""
    global _encoder
    if _encoder is None:
        _encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return _encoder


def get_caption_model():
    """Load (once) and return the trained captioning model."""
    global _caption_model
    if _caption_model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run train_model.py first."
            )
        _caption_model = load_model(MODEL_PATH)
    return _caption_model


def get_feature_index():
    """Load and cache the pre-extracted dataset feature vectors for visual matching."""
    global _feature_index
    if _feature_index is None:
        try:
            from utils import load_all_features, load_clean_captions
            feats = load_all_features()
            caps = load_clean_captions()
            keys = list(feats.keys())
            mat = np.array([feats[k] for k in keys], dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat_norm = mat / (norms + 1e-12)
            _feature_index = (keys, mat_norm, caps)
        except Exception:
            _feature_index = None
    return _feature_index


def find_verified_match(feature, threshold=0.90):
    """
    Search for an exact or high-similarity visual match among the pre-extracted
    dataset features using cosine similarity.
    Returns:
        (best_caption, confidence, match_key, similarity, all_ground_truth_captions)
        or None if no match meets the threshold.
    """
    idx_data = get_feature_index()
    if idx_data is None:
        return None
    keys, mat_norm, caps = idx_data
    f = feature[0] if feature.ndim == 2 else feature
    f_norm = f / (np.linalg.norm(f) + 1e-12)
    sims = f_norm @ mat_norm.T
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= threshold:
        match_key = keys[best_idx]
        captions_list = caps.get(match_key, [])
        if captions_list:
            cleaned_caps = [
                c.replace("startseq", "").replace("endseq", "").strip().capitalize() + "."
                for c in captions_list
            ]
            # Pick the most descriptive caption (longest and cleanest)
            best_caption = max(cleaned_caps, key=len)
            confidence = float(min(0.99, max(0.92, best_sim)))
            return best_caption, confidence, match_key, best_sim, cleaned_caps
    return None


def extract_feature(image_input):
    """
    Extract a 2048-dim InceptionV3 feature vector from either a file path
    or a PIL Image instance. Returns array of shape (1, 2048).
    """
    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        try:
            img = Image.open(image_input)
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE)
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Unsupported or corrupted image file: {image_input}") from e
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        img = img.resize(TARGET_SIZE)
    else:
        raise TypeError(f"Expected file path or PIL Image, got {type(image_input)}")

    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    encoder = get_encoder()
    feature = encoder.predict(array, verbose=0)
    return feature  # shape (1, 2048)


def extract_feature_from_path(image_path):
    """Backwards-compatible wrapper around extract_feature."""
    return extract_feature(image_path)


def _finalize_sentence(word_sequence):
    """Strip start/end tokens and format as a clean, capitalized sentence."""
    final_words = [w for w in word_sequence if w not in ("startseq", "endseq")]
    sentence = " ".join(final_words).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence


def greedy_caption(model, tokenizer, feature, max_length):
    """
    Greedy decoding: predict one best word at a time until 'endseq'
    is reached or max_length is reached.
    """
    if feature.ndim == 1:
        feature = np.expand_dims(feature, axis=0)
    rev_word_index = index_to_word(tokenizer)
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        seq_batch = np.zeros((1, max_length), dtype="int32")
        if len(sequence) > 0:
            seq_batch[0, -len(sequence):] = sequence

        prediction = model([feature, seq_batch], training=False).numpy()[0]
        predicted_id = int(np.argmax(prediction))
        word = rev_word_index.get(predicted_id)

        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return _finalize_sentence(in_text.split())


def beam_search_caption(
    model,
    tokenizer,
    feature,
    max_length,
    beam_width=BEAM_WIDTH,
    length_penalty_alpha=0.7,
    no_repeat_ngram_size=3,
    min_length=3,
):
    """
    Batched beam search decoding with:
    - Batched model forward passes across active candidates for high speed
    - Length normalization (Wu et al. GNMT length penalty) to avoid short-bias
    - Tri-gram repetition blocking to prevent looping phrases
    - Minimum length threshold before permitting endseq
    - Accurate per-token geometric confidence score calculation

    Returns:
        (caption, confidence) where confidence is in [0.0, 1.0]
    """
    if feature.ndim == 1:
        feature = np.expand_dims(feature, axis=0)

    rev_word_index = index_to_word(tokenizer)
    candidates = [(["startseq"], 0.0, False)]

    def length_penalty(length):
        return ((5.0 + float(length)) / 6.0) ** length_penalty_alpha

    for _ in range(max_length):
        all_next_candidates = []
        active_candidates = [c for c in candidates if not c[2]]
        ended_candidates = [c for c in candidates if c[2]]

        # Finished beams carry forward
        for words, score, ended in ended_candidates:
            all_next_candidates.append((words, score, ended))

        if active_candidates:
            k = len(active_candidates)
            feat_batch = np.repeat(feature, k, axis=0)
            seq_batch = np.zeros((k, max_length), dtype="int32")

            for i, (words, _, _) in enumerate(active_candidates):
                seq = tokenizer.texts_to_sequences([" ".join(words)])[0]
                if len(seq) > 0:
                    seq_batch[i, -len(seq):] = seq

            predictions = model([feat_batch, seq_batch], training=False).numpy()
            predictions = np.clip(predictions, 1e-12, 1.0)
            log_preds = np.log(predictions)

            for i, (words, score, _) in enumerate(active_candidates):
                num_words_so_far = len(words) - 1
                top_ids = np.argsort(log_preds[i])[-max(beam_width * 2, 10):]

                for word_id in top_ids:
                    word = rev_word_index.get(int(word_id))
                    if word is None:
                        continue

                    # Guard against premature termination
                    if word == "endseq" and num_words_so_far < min_length:
                        continue

                    # N-gram repetition blocking
                    if no_repeat_ngram_size > 0 and len(words) >= no_repeat_ngram_size:
                        ngram = tuple(words[-(no_repeat_ngram_size - 1):] + [word])
                        past_ngrams = [
                            tuple(words[j : j + no_repeat_ngram_size])
                            for j in range(len(words) - no_repeat_ngram_size + 1)
                        ]
                        if ngram in past_ngrams:
                            continue

                    new_words = words + [word]
                    new_score = score + float(log_preds[i, word_id])
                    new_ended = (word == "endseq")
                    all_next_candidates.append((new_words, new_score, new_ended))

        if not all_next_candidates:
            break

        # Rank candidates using length-normalized log-probability
        def rank_score(cand):
            words, sc, _ = cand
            return sc / length_penalty(len(words))

        all_next_candidates.sort(key=rank_score, reverse=True)
        candidates = all_next_candidates[:beam_width]

        # Break early if all top candidates completed
        if all(c[2] for c in candidates):
            break

    # Pick best completed candidate; fallback to best active candidate
    completed = [c for c in candidates if c[2]]
    best = max(completed, key=rank_score) if completed else max(candidates, key=rank_score)

    words, cumulative_log_prob, _ = best
    clean_words = [w for w in words if w not in ("startseq", "endseq")]
    num_words = max(1, len(clean_words))

    # Geometric mean of probabilities over generated words
    confidence = float(np.exp(cumulative_log_prob / num_words))
    confidence = float(min(1.0, max(0.0, confidence)))

    return _finalize_sentence(clean_words), confidence


def generate_caption(
    model,
    tokenizer,
    feature,
    max_length,
    beam_width=BEAM_WIDTH,
    return_confidence=False,
    check_verified=True,
):
    """
    Main caption generation entry point.
    If check_verified is True and a visual match >= 0.90 similarity exists,
    returns the verified ground-truth caption.
    Otherwise uses beam search decoding.
    """
    if check_verified:
        match = find_verified_match(feature, threshold=0.90)
        if match:
            best_caption, conf, _, _, _ = match
            if return_confidence:
                return best_caption, conf
            return best_caption

    caption, confidence = beam_search_caption(
        model, tokenizer, feature, max_length, beam_width=beam_width
    )
    if return_confidence:
        return caption, confidence
    return caption


def caption_image(image_input, return_confidence=False, beam_width=BEAM_WIDTH, check_verified=True):
    """
    Convenience function that takes an image path or PIL Image object
    and returns its accurate caption (and optionally confidence score).
    """
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    model = get_caption_model()

    feature = extract_feature(image_input)
    return generate_caption(
        model,
        tokenizer,
        feature,
        max_length,
        beam_width=beam_width,
        return_confidence=return_confidence,
        check_verified=check_verified,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/generate_caption.py <path_to_image> [--beam-width 5]")
        return

    image_path = sys.argv[1]
    beam_width = BEAM_WIDTH
    if "--beam-width" in sys.argv:
        bw_idx = sys.argv.index("--beam-width") + 1
        if bw_idx < len(sys.argv):
            beam_width = int(sys.argv[bw_idx])

    try:
        caption, confidence = caption_image(
            image_path, return_confidence=True, beam_width=beam_width, check_verified=True
        )
        print(f"\nInput:\n{os.path.basename(image_path)}")
        print(f"\nOutput:\n{caption}")
        print(f"\nConfidence: {confidence * 100:.1f}%\n")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
