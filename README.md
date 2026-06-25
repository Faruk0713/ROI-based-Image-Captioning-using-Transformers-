# Hindi–English Visual Genome Image Captioning (ROI-based)

This project implements an image captioning pipeline for the **Hindi Visual Genome** dataset, generating captions in both **English** and **Hindi** from region-of-interest (ROI) crops of images. It covers the full pipeline — feature extraction, tokenizer training, sequence-to-sequence caption generation with a Transformer decoder, and BLEU-based evaluation.

## 🧠 Pipeline Overview

```
Raw Images + Bounding Boxes (Hindi Visual Genome)
        │
        ▼
ROI Cropping + ResNet-50 Feature Extraction  (.npy / .pkl)
        │
        ▼
SentencePiece Tokenizer Training (Hindi & English)
        │
        ▼
Transformer Decoder Training (per-language)
        │
        ▼
Greedy Decoding / Caption Inference
        │
        ▼
SacreBLEU Evaluation
```

## 📁 Repository Contents

| Notebook | Purpose |
|---|---|
| `ROI_Feature_Extraction_Split_I.ipynb` | Extracts ResNet-50 features from training-set ROI crops, validates `.npy` outputs, and packs them into a consolidated `.pkl` feature dictionary for fast loading. |
| `ROI_Feature_Extraction_Split_II.ipynb` | Extracts ROI features for the test, dev/validation, and challenge test splits, then zips/backs them up to Drive. |
| `Hindi_Tokenizer_Training.ipynb` | Cleans raw Hindi captions and trains SentencePiece unigram tokenizers (vocab sizes 3000 and 5000). |
| `Hindi_Caption_Training_Script_ROI.ipynb` | Trains a Transformer-decoder captioning model on Hindi captions using ROI features, with inference and reference-merging utilities. |
| `English_Caption_Training_Script_ROI.ipynb` | Trains the same Transformer-decoder architecture on English captions using ROI features, with inference and reference-merging utilities. |
| `SacreBLEU_Evaluation_Script_Hin_Eng.ipynb` | Computes SacreBLEU scores (1–2, 1–3, and 1–4 gram) for generated vs. reference captions in both languages. |

> **Note:** The English SentencePiece tokenizer training notebook is not yet included in this repo — it will be added soon. References to `english_unigram_4500.model` in the training/inference notebooks assume this tokenizer has been trained separately, following the same approach as the Hindi tokenizer notebook.

## 🛠️ Key Tools & Frameworks

**Feature Extraction**
- **PyTorch** + **torchvision** — `ResNet-50` (ImageNet-pretrained, final FC layer stripped) used as a fixed CNN feature extractor over cropped ROI regions, producing 2048-dimensional feature vectors.
- **Pillow (PIL)** — image loading and ROI cropping based on bounding-box coordinates.
- **Pandas** — parsing the tab-separated Hindi Visual Genome annotation files (`image_id`, bounding box, English/Hindi captions).
- **NumPy** — feature vector storage (`.npy`) and pooling of region-level features.

**Tokenization**
- **SentencePiece** — unigram-model subword tokenizers trained directly on cleaned caption text (Hindi: 3000/5000 vocab; English: 4500 vocab, pending upload). Provides `bos`/`eos`/`pad` token handling and lossless encode/decode for both Devanagari and Latin scripts.

**Model & Training**
- **PyTorch (`torch.nn`)** — custom **Transformer Decoder** architecture:
  - Sinusoidal `PositionalEncoding` module
  - `nn.TransformerDecoder` (multi-head self-attention + cross-attention)
  - A linear `image_proj` layer mapping the 2048-d ResNet feature into the decoder's memory/cross-attention input (acting as a single-vector "encoder" output)
  - Causal (subsequent-token) masking for autoregressive decoding
- **Adam optimizer** with gradient clipping and padding-aware `CrossEntropyLoss`.
- **tqdm** — training/extraction progress bars.

**Inference**
- Greedy autoregressive decoding starting from the `<bos>` token, using the same SentencePiece model for de-tokenization into readable text.

**Evaluation**
- **SacreBLEU** (`corpus_bleu` and the `BLEU` metric class) — standardized BLEU scoring against reference captions, reported at multiple n-gram orders (1–2, 1–3, 1–4 gram) for both the English and Hindi caption outputs.
- **NLTK** (`corpus_bleu`) — imported for supplementary BLEU comparison in the tokenizer notebook.

**Environment**
- Built and run on **Google Colab** (T4 GPU runtime) with Google Drive used for dataset/model/checkpoint storage.

## 📊 Dataset

[Hindi Visual Genome](http://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2987) — image regions paired with parallel English and Hindi captions and bounding-box annotations, split into train/dev/test/challenge-test sets.

## 🚀 Usage Workflow

1. **Extract ROI features** — run `ROI_Feature_Extraction_Split_I.ipynb` (train set) and `ROI_Feature_Extraction_Split_II.ipynb` (test/dev/challenge sets) to produce `.npy`/`.pkl` ResNet-50 feature files for each ROI crop.
2. **Train tokenizers** — run `Hindi_Tokenizer_Training.ipynb` to produce the Hindi SentencePiece model(s). Train an equivalent English SentencePiece model (`english_unigram_4500.model`) the same way.
3. **Train captioning models** — run `Hindi_Caption_Training_Script_ROI.ipynb` and `English_Caption_Training_Script_ROI.ipynb` to train the Transformer decoder for each language on the extracted ROI features.
4. **Generate captions** — use the `generate_caption()` inference utility within each training notebook to produce captions for the test/challenge sets.
5. **Evaluate** — run `SacreBLEU_Evaluation_Script_Hin_Eng.ipynb` on the merged reference/generated caption JSON files to compute SacreBLEU scores.

## 📦 Requirements

```
torch
torchvision
sentencepiece
sacrebleu
nltk
tensorflow / keras   # used in tokenizer notebook for VGG16-related imports
numpy
pandas
pillow
tqdm
```

## 📌 Notes

- Feature extraction assumes a CUDA-enabled GPU runtime for ResNet-50 inference at reasonable speed.
- Image IDs are matched across captions, bounding boxes, and `.npy`/`.pkl` feature files via the prefix before the first underscore in each filename (`{image_id}_{row_index}.npy`).
- The current English/Hindi captioning models use a single pooled 2048-d image feature per ROI rather than a full sequence of region features, fed into the Transformer decoder as a single-token memory/cross-attention input.
