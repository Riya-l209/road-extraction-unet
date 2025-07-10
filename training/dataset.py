# dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image filename
        image_filename = self.images[idx]

        # Build full path to the image
        image_path = os.path.join(self.image_dir, image_filename)

        # Derive corresponding mask filename
        filename = os.path.splitext(image_filename)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, filename)

        # Safety check
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for: {mask_path}")
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
