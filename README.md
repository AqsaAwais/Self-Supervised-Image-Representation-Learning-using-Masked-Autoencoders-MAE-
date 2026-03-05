# Self-Supervised-Image-Representation-Learning-using-Masked-Autoencoders-MAE-
PyTorch implementation of Masked Autoencoder (MAE) for self-supervised image representation learning on TinyImageNet. Asymmetric ViT encoder-decoder, 75% masking, training on Kaggle GPU, with Gradio app for image upload and reconstruction.
# Masked Autoencoder (MAE) on TinyImageNet

Self-supervised image representation learning using a Masked Autoencoder (MAE) with an asymmetric Vision Transformer (ViT) encoder-decoder, trained on TinyImageNet and demoed via Gradio.

## Overview

- **Encoder:** ViT-Base (B/16) — processes only 25% visible patches (no mask tokens).  
- **Decoder:** ViT-Small (S/16) — reconstructs full image from encoder latents + learnable mask tokens.  
- **Objective:** Reconstruct masked 75% of patches; loss (MSE) computed only on masked patches.  
- **Dataset:** TinyImageNet (224×224).  
- **Stack:** PyTorch (base layers), optional training on Kaggle (GPU T4 x2), Gradio for inference demo.

## Features

- Patchification and random masking (configurable ratio, default 75% masked).  
- ViT-Base encoder (768-d, 12 layers, 12 heads) and ViT-Small decoder (384-d, 12 layers, 6 heads).  
- Training with AdamW, cosine LR schedule, mixed precision (AMP), gradient clipping.  
- Visualization: masked input, reconstruction, and original for ≥5 samples.  
- Quantitative evaluation: PSNR and SSIM on reconstructed images.  
- Gradio app: image upload, mask-ratio slider, real-time reconstruction.

## Repository structure
