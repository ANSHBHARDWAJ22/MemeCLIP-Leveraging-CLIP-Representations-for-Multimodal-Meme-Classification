# MemeCLIP-Leveraging-CLIP-Representations-for-Multimodal-Meme-Classification





![Screenshot 2025-05-30 144723](https://github.com/user-attachments/assets/35d56737-2435-4790-af94-d38e8050dec1)

# How MemeCLIP is Different from Original CLIP and Prior Models
MemeCLIP is a lightweight and efficient extension of the CLIP architecture, specifically designed for understanding memes. Here's how it differs from earlier models:

Previous Approaches
MOMENTA (2021)
Used CLIP's image and text features but added extra processing like extracting regions of interest from images and named entities from text. It relied heavily on cross-modal attention, making the model complex and slow.

HateCLIPper (2022)
Improved how image and text features interact and fuse, but it was still prone to overfitting, especially with smaller or imbalanced datasets.

ISSUES (2023)
Tried mapping visual signals directly into text space using textual inversion techniques. Smart, but the architecture was heavy and required more resources.

Image Captioning + NLP Models (2023)
Generated captions first and then ran NLP models on them. This two-step approach added extra processing and model dependency.

# What MemeCLIP Does Differently
It doesn't rely on external captioning models or complex augmentations.

Uses frozen CLIP image and text encoders to directly extract rich representations.

Introduces Feature Adapters with residual connections to reduce overfitting while preserving pre-trained knowledge.

Uses a cosine classifier, which works better on imbalanced datasets by comparing feature directions rather than magnitudes.

Initializes the classifier with semantic-aware weights, improving performance without extra training cost.
