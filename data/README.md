# 📂 Road Extraction Dataset (Custom Sample)

This is a sample dataset used for training and testing a U-Net model to extract roads from satellite imagery.

## 📁 Folder Structure

data/
└── raw/
    ├── images/   # Contains 6 satellite TIFF images
    └── masks/    # Contains 6 corresponding binary road masks in TIFF format

## 📷 Image Info

- Format: .tiff
- Resolution: 1500 x 1500 pixels (or as per actual)
- Channels:
  - Satellite images: RGB (3-channel)
  - Masks: Grayscale (1-channel), where:
    - 255 = road (white)
    - 0 = background (black)

## 🔁 Naming Convention

- Each image in images/ has a matching mask in masks/
  - Example: satellite_01.tiff → mask_01.tiff

## 👤 Maintainer

Vinayak – Dataset Manager (Branch: vinayak-datamanage)

