"""
diagnose_keys.py
-----------------
Quick standalone diagnostic: checks how many keys overlap between
captions_clean.pkl and the feature .npy files, and prints sample keys
from each so you can see exactly why they aren't matching.

Run from the project root:
    python src/diagnose_keys.py
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(MODELS_DIR, "features")
CLEAN_CAPTIONS_PATH = os.path.join(MODELS_DIR, "captions_clean.pkl")


def main():
    if not os.path.isfile(CLEAN_CAPTIONS_PATH):
        print(f"[ERROR] {CLEAN_CAPTIONS_PATH} not found. Run preprocess_captions.py first.")
        return
    if not os.path.isdir(FEATURES_DIR):
        print(f"[ERROR] {FEATURES_DIR} not found. Run feature_extraction.py first.")
        return

    with open(CLEAN_CAPTIONS_PATH, "rb") as f:
        clean_captions = pickle.load(f)

    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]
    feature_keys = {os.path.splitext(f)[0] for f in feature_files}
    caption_keys = set(clean_captions.keys())

    print(f"Total caption entries:  {len(caption_keys)}")
    print(f"Total feature (.npy) files: {len(feature_keys)}")

    print("\nSample caption keys (raw, as stored in captions_clean.pkl):")
    for k in list(caption_keys)[:5]:
        print(f"   {repr(k)}")

    print("\nSample feature keys (raw, .npy filenames minus extension):")
    for k in list(feature_keys)[:5]:
        print(f"   {repr(k)}")

    # Direct overlap (no normalization)
    direct_overlap = caption_keys & feature_keys
    print(f"\nDirect overlap (no normalization): {len(direct_overlap)}")

    # Overlap after stripping a trailing image extension from caption keys
    normalized_caption_keys = {os.path.splitext(k)[0] for k in caption_keys}
    normalized_overlap = normalized_caption_keys & feature_keys
    print(f"Overlap after stripping extensions from caption keys: {len(normalized_overlap)}")

    if len(feature_keys) == 0:
        print("\n[DIAGNOSIS] models/features/ has NO .npy files at all.")
        print("Feature extraction did not run or did not complete. Re-run:")
        print("   python src/feature_extraction.py")
    elif len(direct_overlap) > 0:
        print("\n[DIAGNOSIS] Keys already match directly. The issue is likely")
        print("elsewhere (e.g. an old cached train_model.py). No extension")
        print("mismatch detected.")
    elif len(normalized_overlap) > 0:
        print("\n[DIAGNOSIS] Extension mismatch confirmed: caption keys carry")
        print("an image extension (e.g. '.jpg') that feature keys don't have.")
        print("This is exactly what the train_model.py fix normalizes.")
        print("Make sure your src/train_model.py actually contains the fix.")
    else:
        print("\n[DIAGNOSIS] Keys don't overlap even after stripping extensions.")
        print("This points to something else — e.g. filenames were renamed,")
        print("or feature_extraction.py ran against a different/partial image")
        print("folder than preprocess_captions.py's token file describes.")
        print("Compare a few keys above by eye to spot the real difference.")


if __name__ == "__main__":
    main()
