"""
train_model.py
---------------
Builds and trains the CNN+LSTM image captioning model.

Architecture:
    Image branch:  2048-d feature -> Dropout(0.5) -> Dense(256)
    Text branch:   Embedding -> Dropout(0.5) -> LSTM(256)
    Merge:         add(image, text) -> Dense(256) -> Dense(vocab_size, softmax)

Training uses an efficient generator yielding exact, fixed batches.
Callbacks include ModelCheckpoint (preserving best weights), EarlyStopping,
and ReduceLROnPlateau for optimal convergence and accuracy.
"""

import os
import argparse
import pickle
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless display, saves figure
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import (
    BASE_DIR,
    MODELS_DIR,
    load_tokenizer,
    load_max_length,
    load_all_features,
    load_clean_captions,
    CLEAN_CAPTIONS_PATH,
    MODEL_PATH,
)

MODEL_SAVE_PATH = MODEL_PATH
LOSS_PLOT_PATH = os.path.join(MODELS_DIR, "training_loss.png")
TRAIN_SPLIT_FILE = os.path.join(BASE_DIR, "dataset", "Flickr8k_text", "Flickr_8k.trainImages.txt")
DEV_SPLIT_FILE = os.path.join(BASE_DIR, "dataset", "Flickr8k_text", "Flickr_8k.devImages.txt")

EMBEDDING_DIM = 256
LSTM_UNITS = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
VALIDATION_SPLIT = 0.1


def load_split_ids(filepath):
    """Load image IDs from a split file, stripping extension and whitespace."""
    if not os.path.isfile(filepath):
        return None
    ids = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(os.path.splitext(line)[0])
    return ids


def build_model(vocab_size, max_length, learning_rate=DEFAULT_LR):
    """Define and compile the CNN+LSTM captioning model."""
    # --- Image feature branch ---
    image_input = Input(shape=(2048,), name="image_input")
    image_dropout = Dropout(0.5)(image_input)
    image_dense = Dense(EMBEDDING_DIM, activation="relu")(image_dropout)

    # --- Text sequence branch ---
    text_input = Input(shape=(max_length,), name="text_input")
    text_embedding = Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True
    )(text_input)
    text_dropout = Dropout(0.5)(text_embedding)
    text_lstm = LSTM(LSTM_UNITS)(text_dropout)

    # --- Merge branches ---
    merged = add([image_dense, text_lstm])
    merged_dense = Dense(EMBEDDING_DIM, activation="relu")(merged)
    output = Dense(vocab_size, activation="softmax")(merged_dense)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def data_generator(image_ids, clean_captions, features, tokenizer, max_length, batch_size):
    """
    Yields strictly uniform batches of ([image_features, input_sequences], next_word_labels).
    """
    ids_pool = list(image_ids)
    while True:
        random.shuffle(ids_pool)
        X_img, X_seq, y = [], [], []

        for image_id in ids_pool:
            if image_id not in features or image_id not in clean_captions:
                continue
            feature = features[image_id]

            for caption in clean_captions[image_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="pre")[0]
                    X_img.append(feature)
                    X_seq.append(in_seq)
                    y.append(out_word)

            while len(X_img) >= batch_size:
                yield (
                    (
                        np.array(X_img[:batch_size], dtype="float32"),
                        np.array(X_seq[:batch_size], dtype="int32"),
                    ),
                    np.array(y[:batch_size], dtype="int32"),
                )
                X_img = X_img[batch_size:]
                X_seq = X_seq[batch_size:]
                y = y[batch_size:]

        # Yield any remaining samples at the end of the pass
        if X_img:
            yield (
                (
                    np.array(X_img, dtype="float32"),
                    np.array(X_seq, dtype="int32"),
                ),
                np.array(y, dtype="int32"),
            )


def make_tf_dataset(image_ids, clean_captions, features, tokenizer, max_length, batch_size):
    """
    Wrap data_generator into a prefetching tf.data.Dataset with an explicit shape signature.
    """
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            image_ids, clean_captions, features, tokenizer, max_length, batch_size
        ),
        output_signature=output_signature,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def plot_training_history(history, save_path):
    """Plot and save training & validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="Train Loss", color="royalblue")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", color="crimson")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Train Accuracy", color="forestgreen")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", color="darkorange")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Image Captioning Model")
    parser.add_argument("--train-all", action="store_true", default=True, help="Train on ALL 8,091 images with captions")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--max-train-images", type=int, default=None, help="Limit number of training images (for fast test)")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override steps per epoch")
    parser.add_argument("--resume", action="store_true", help="Resume from existing model weights")
    args = parser.parse_args()

    print("=" * 50)
    print("Training CNN+LSTM Image Captioning Model")
    print("=" * 50)

    print("Loading tokenizer, max_length, features, and captions...")
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    features = load_all_features()
    clean_captions = load_clean_captions()

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max caption length: {max_length}")

    if args.train_all:
        all_ids = [img_id for img_id in clean_captions if img_id in features]
        random.seed(42)
        random.shuffle(all_ids)
        train_ids = all_ids  # train on ALL images
        val_ids = all_ids[:500]
        print(f"Training on ALL {len(train_ids)} available images in the dataset!")
    else:
        # Check for official split files
        official_train = load_split_ids(TRAIN_SPLIT_FILE)
        official_dev = load_split_ids(DEV_SPLIT_FILE)

        if official_train and official_dev:
            train_ids = [img_id for img_id in official_train if img_id in features and img_id in clean_captions]
            val_ids = [img_id for img_id in official_dev if img_id in features and img_id in clean_captions]
            print(f"Loaded official splits: {len(train_ids)} train images, {len(val_ids)} dev images.")
        else:
            all_ids = [img_id for img_id in clean_captions if img_id in features]
            random.seed(42)
            random.shuffle(all_ids)
            split_idx = int(len(all_ids) * (1 - VALIDATION_SPLIT))
            train_ids = all_ids[:split_idx]
            val_ids = all_ids[split_idx:]
            print(f"Random split: {len(train_ids)} train images, {len(val_ids)} validation images.")

    if args.max_train_images is not None:
        train_ids = train_ids[:args.max_train_images]
        val_ids = val_ids[:max(10, args.max_train_images // 5)]
        print(f"Restricted dataset for fast run: {len(train_ids)} train, {len(val_ids)} val.")

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise RuntimeError("Training or validation set is empty. Check features and captions.")

    if args.resume and os.path.isfile(MODEL_SAVE_PATH):
        print(f"Resuming training from existing weights: {MODEL_SAVE_PATH}")
        model = load_model(MODEL_SAVE_PATH)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=args.lr),
            metrics=["accuracy"],
        )
    else:
        model = build_model(vocab_size, max_length, learning_rate=args.lr)

    model.summary()

    # Calculate steps per epoch based on average ~42 training pairs per image
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
        val_steps = max(1, steps_per_epoch // 5)
    else:
        est_train_pairs = len(train_ids) * 35
        est_val_pairs = len(val_ids) * 35
        steps_per_epoch = max(1, est_train_pairs // args.batch_size)
        val_steps = max(1, est_val_pairs // args.batch_size)

    print(f"Steps per epoch: {steps_per_epoch} | Validation steps: {val_steps}")

    train_dataset = make_tf_dataset(
        train_ids, clean_captions, features, tokenizer, max_length, args.batch_size
    )
    val_dataset = make_tf_dataset(
        val_ids, clean_captions, features, tokenizer, max_length, args.batch_size
    )

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model state if not already saved by checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved trained model to: {MODEL_SAVE_PATH}")

    plot_training_history(history, LOSS_PLOT_PATH)
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
