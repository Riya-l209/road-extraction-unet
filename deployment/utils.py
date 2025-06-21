import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

from model.unet import UNet  # Import your U-Net model class

# Load and initialize the model
def load_model(model_path="saved_models/best_model.pth", device="cpu"):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path, img_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Shape: [1, 3, H, W]

# Postprocess the predicted mask
def postprocess_mask(pred_mask, threshold=0.5):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = pred_mask.squeeze().cpu().detach().numpy()
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
    return pred_mask

# Save the predicted mask as an image
def save_mask(mask, output_path="static/predicted_mask.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)
    return output_path
