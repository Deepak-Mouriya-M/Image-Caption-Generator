# Image Caption Generator Using Deep Learning (CNN + LSTM)

A B.Tech AI & Data Science mini project that automatically generates
natural-language captions for images, combining a CNN image encoder
(InceptionV3) with an LSTM language decoder, wrapped in a Streamlit
web application.

## 1. Project Overview

Given an input image, the system extracts high-level visual features
using a pretrained InceptionV3 network, then feeds those features
into an LSTM-based decoder that generates a caption word by word.

```
Image
   │
   ▼
InceptionV3 (Encoder)
   │
2048 Feature Vector
   │
   ▼
Embedding + LSTM (Decoder)
   │
   ▼
Generated Caption
```

Example:
- Input: a photo of a dog running in grass
- Output: "A brown dog is running through the grass."

## 2. Objectives

- Apply CNN-based transfer learning (InceptionV3) for image feature extraction
- Build a sequence model (LSTM) for natural language generation
- Combine both into an end-to-end encoder-decoder captioning architecture
- Deploy the trained model behind an interactive Streamlit interface
- Evaluate caption quality using the BLEU metric

## 3. CNN-LSTM Architecture

**Image branch (encoder):**
- Input: 2048-d InceptionV3 feature vector
- Dropout(0.5)
- Dense(256, relu)

**Text branch (decoder):**
- Embedding layer (vocab_size → 256), mask_zero=True
- Dropout(0.5)
- LSTM(256)

**Merge:**
- add(image_branch, text_branch)
- Dense(256, relu)
- Dense(vocab_size, softmax)

Loss: Sparse Categorical Crossentropy · Optimizer: Adam

## 4. Folder Structure

```
Image-Caption-Generator/
│
├── dataset/
│   ├── Flickr8k_Dataset/        # ~8091 .jpg images
│   └── Flickr8k_text/           # caption + split files
│
├── models/
│   ├── features/                # cached InceptionV3 features (.npy)
│   ├── tokenizer.pkl
│   ├── max_length.pkl
│   ├── captions_clean.pkl
│   ├── caption_model.keras
│   └── training_loss.png
│
├── src/
│   ├── check_dataset.py
│   ├── preprocess_captions.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── generate_caption.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 5. Installation

Requires Python 3.11. No virtual environment needed — install directly:

```
pip install -r requirements.txt
```

## 6. Dataset Setup

Download the **Flickr8k** dataset and place it as:

```
dataset/Flickr8k_Dataset/   -> all .jpg images
dataset/Flickr8k_text/      -> Flickr8k.token.txt, Flickr_8k.trainImages.txt,
                                Flickr_8k.devImages.txt, Flickr_8k.testImages.txt
```

## 7. How to Run

Run each step from the project root, in order:

```
# 1. Verify the dataset is complete
python src/check_dataset.py

# 2. Clean captions, build tokenizer
python src/preprocess_captions.py

# 3. Extract InceptionV3 features for every image (can take a while)
python src/feature_extraction.py

# 4. Train the CNN+LSTM model
python src/train_model.py

# 5. (Optional) Evaluate with BLEU score
python src/evaluate_model.py

# 6. Generate a caption for a single new image from the command line
python src/generate_caption.py path/to/your_image.jpg

# 7. Launch the web app
streamlit run app.py
```

## 8. Sample Output

```
Input:
dog.jpg

Output:
A brown dog is running through the grass.
```

## 9. Evaluation

`evaluate_model.py` computes BLEU-1 through BLEU-4 scores over a
sample of test images by comparing generated captions against the
ground-truth reference captions, and prints the full Keras model
summary.

## 10. Error Handling

The pipeline validates and reports clear messages for:
- missing dataset folders/files
- missing feature (.npy) files
- missing tokenizer / max_length / model files
- unsupported or corrupted image uploads

## 11. Future Scope

- Replace greedy decoding with beam search for higher-quality captions
- Add an attention mechanism (e.g. Show, Attend and Tell) over CNN feature maps
- Fine-tune InceptionV3 on the target dataset instead of using it frozen
- Train on larger datasets (Flickr30k, MS-COCO) for better generalization
- Add multilingual caption generation
