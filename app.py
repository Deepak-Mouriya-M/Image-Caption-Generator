"""
app.py
------
Streamlit web application for the Image Caption Generator.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Make src/ importable
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH_PATH, load_tokenizer, load_max_length
from generate_caption import (
    get_caption_model,
    get_encoder,
    extract_feature,
    generate_caption,
    find_verified_match,
    beam_search_caption,
)

# ---------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }
    .caption-card {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .caption-text {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1e3a8a;
        margin: 1rem 0;
        line-height: 1.5;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #2563eb;
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .match-tag {
        display: inline-block;
        background-color: #10b981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🖼️ Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image and let a deep CNN + LSTM model describe it accurately</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Resource caching for high performance
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_app_resources():
    """Load model, tokenizer, max_length, and encoder once and cache in memory."""
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    model = get_caption_model()
    encoder = get_encoder()
    return model, tokenizer, max_length, encoder


# ---------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------

missing_artifacts = []
if not os.path.isfile(MODEL_PATH):
    missing_artifacts.append("trained model (models/caption_model.keras)")
if not os.path.isfile(TOKENIZER_PATH):
    missing_artifacts.append("tokenizer (models/tokenizer.pkl)")
if not os.path.isfile(MAX_LENGTH_PATH):
    missing_artifacts.append("max_length (models/max_length.pkl)")

if missing_artifacts:
    st.error(
        "The following required artifacts are missing:\n\n"
        + "\n".join(f"- {item}" for item in missing_artifacts)
        + "\n\nPlease run `preprocess_captions.py` and `train_model.py` before using the app."
    )
    st.stop()

# ---------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    beam_width = st.slider("Beam Width", min_value=1, max_value=7, value=5, help="Higher beam widths explore more word combinations.")
    enable_matching = st.checkbox("Enable Verified Visual Matching", value=True, help="Matches against verified visual ground truth library when available.")

# ---------------------------------------------------------------------
# Main interaction
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
    except UnidentifiedImageError:
        st.error("This file could not be read as an image. Please upload a valid JPG, PNG, or WEBP image.")
        st.stop()

    st.image(image, caption="Uploaded Image")

    if st.button("Generate Caption", type="primary"):
        with st.spinner("Extracting visual features and generating caption..."):
            try:
                model, tokenizer, max_length, encoder = load_app_resources()

                # In-memory feature extraction directly from the PIL image
                feature = extract_feature(image)

                verified_match = find_verified_match(feature, threshold=0.90) if enable_matching else None

                if verified_match:
                    caption, confidence, match_key, sim, all_caps = verified_match
                    confidence_pct = confidence * 100
                    tag_html = f'<div class="match-tag">✓ Verified Visual Match ({sim*100:.1f}%)</div>'
                else:
                    caption, confidence = beam_search_caption(
                        model=model,
                        tokenizer=tokenizer,
                        feature=feature,
                        max_length=max_length,
                        beam_width=beam_width,
                    )
                    confidence_pct = confidence * 100
                    tag_html = '<div style="font-size: 0.95rem; color: #475569; font-weight: 500;">Generated Caption</div>'
                    all_caps = None

                # Print to terminal for review
                print(f"\n[Inference] Image: {uploaded_file.name}")
                print(f"  Caption:    {caption}")
                print(f"  Confidence: {confidence_pct:.1f}%\n")

                st.markdown(
                    f"""
                    <div class="caption-card">
                        {tag_html}
                        <div class="caption-text">"{caption}"</div>
                        <div class="confidence-badge">Confidence: {confidence_pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if all_caps and len(all_caps) > 1:
                    with st.expander("View Alternative Human Ground-Truth Captions"):
                        for idx, alt_cap in enumerate(all_caps, 1):
                            st.write(f"**{idx}.** {alt_cap}")

            except FileNotFoundError as e:
                st.error(f"Missing required artifact: {e}")
            except Exception as e:
                st.error(f"Error while generating caption: {e}")
else:
    st.info("Upload a JPG, PNG, or WEBP image to get started.")
