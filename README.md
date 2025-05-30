# MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification

![MemeCLIP Architecture](https://github.com/user-attachments/assets/35d56737-2435-4790-af94-d38e8050dec1)

## How MemeCLIP is Different from Original CLIP and Prior Models

MemeCLIP is a lightweight and efficient extension of the CLIP architecture, specifically designed for understanding memes. Here's how it differs from earlier models:

### Previous Approaches

#### MOMENTA (2021)
- Used CLIP's image and text features.
- Applied extra processing like extracting regions of interest (ROI) from images and named entities from text.
- Relied heavily on cross-modal attention, making it complex and computationally heavy.

#### HateCLIPper (2022)
- Improved the fusion and interaction of image-text features.
- Still prone to overfitting, especially on smaller or imbalanced datasets.

#### ISSUES (2023)
- Mapped visual cues into the textual space using textual inversion techniques.
- Innovative but introduced a heavier architecture with more resource requirements.

#### Image Captioning + NLP Models (2023)
- Generated image captions first, then passed them through NLP models.
- Dependent on extra models and multiple processing steps.

---

## What MemeCLIP Does Differently

- Does **not rely** on external captioning models or complex preprocessing.
- Uses **frozen CLIP image and text encoders** to extract rich multimodal representations directly.
- Introduces **Feature Adapters** with **residual connections** to:
  - Retain pre-trained knowledge.
  - Reduce overfitting on small datasets.
- Uses a **cosine classifier**, which performs better in imbalanced classification settings by comparing feature directions instead of magnitudes.
- Employs **semantic-aware initialization** for the classifier, enhancing performance without extra training cost.

---

## 🔧 Architecture Overview

MemeCLIP is a lightweight, CLIP-based architecture optimized for multimodal meme classification. It leverages CLIP’s strong pre-trained encoders but introduces a few key components to adapt to the unique challenges of meme data — such as sarcasm, contrast, and noisy supervision.

---

### 1. CLIP Encoders (Frozen)

We use the standard CLIP image and text encoders:

- `EI(I)` → image embedding `FI ∈ ℝ⁷⁶⁸`
- `ET(T)` → text embedding `FT ∈ ℝ⁷⁶⁸`

These encoders are **frozen** during training to retain their robust multimodal understanding.

---

### 2. Modality-Specific Projections

Memes often have **contrasting visual and textual content**. Instead of forcing both into the same embedding space directly, we use **separate linear layers** for each:

- `F_proj_I = L_proj_I(FI)`
- `F_proj_T = L_proj_T(FT)`

This maps both to `ℝ¹⁰²⁴`, aligning them with CLIP's final hidden dimension, while keeping modality-specific signals intact.

---

### 3. Feature Adapters + Residual Blending

To avoid overfitting on small datasets, we introduce **Feature Adapters** — small trainable modules that refine the features without destroying pre-trained knowledge.

Each adapter output is blended with the original projection using a residual ratio `α`:

- `F_final_I = α × Adapter_I(F_proj_I) + (1 − α) × F_proj_I`
- `F_final_T = α × Adapter_T(F_proj_T) + (1 − α) × F_proj_T`

This enables **controlled fine-tuning**, preserving generalization while adapting to meme-specific data.

---

### 4. Multimodal Fusion

Instead of using heavy cross-modal attention (like in MOMENTA), we fuse image and text features with a simple but effective **element-wise multiplication**:

- `F_MM = F_final_I ⊙ F_final_T`

This yields a rich multimodal representation with minimal additional parameters.

---

### 5. Cosine Classifier with Semantic Initialization

We use a **cosine similarity-based classifier**, which is robust under **class imbalance**.

The classifier weights are initialized using CLIP's text encoder with semantic prompts like:

> `"A photo of {CLASS_NAME}"`

This is known as **Semantic-Aware Initialization (SAI)** — helping the classifier generalize better from the start.

---

###  Summary

| Component             | Purpose                                                        |
|-----------------------|----------------------------------------------------------------|
| Frozen CLIP Encoders  | Preserve strong pre-trained visual-linguistic understanding    |
| Linear Projections    | Disentangle modality-specific signals                          |
| Feature Adapters      | Prevent overfitting, retain CLIP knowledge                     |
| Residual Connections  | Blend old + new knowledge safely                               |
| Element-wise Fusion   | Lightweight multimodal representation                          |
| Cosine Classifier     | Handles class imbalance well, no need for retraining           |
| SAI Init              | Uses text semantics to guide classifier from the beginning      |
