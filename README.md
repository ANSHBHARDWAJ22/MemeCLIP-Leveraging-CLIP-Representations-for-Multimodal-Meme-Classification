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

