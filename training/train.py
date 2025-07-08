# train.py
print("🚀 Starting training...")

import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import UNet, RoadDataset 
from losses import CombinedLoss 

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Paths
image_dir = r"C:\\Users\\harsh\\Music\\OneDrive\Desktop\\road-extraction-unet\\all new files\\images"
mask_dir = r"C:\\Users\\harsh\\Music\\OneDrive\Desktop\\road-extraction-unet\\all new files\\masks"

# Dataset
dataset = RoadDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
print(f" Found {len(dataset)} images")

# Train/Val Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# Model, Loss, Optimizer
model = UNet().to(device)
loss_fn = CombinedLoss(alpha=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Helper functions
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(dataloader, desc="🔁 Training"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def calculate_iou(preds, masks):
    preds = (preds.sigmoid() > 0.5).float()
    intersection = (preds * masks).sum((1, 2, 3))
    union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
    return (intersection / (union + 1e-6)).mean().item()

def val_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="🧪 Validating"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
            total_iou += calculate_iou(preds, masks)
    return total_loss / len(dataloader), total_iou / len(dataloader)

# Training Loop
best_loss = float('inf')
best_iou = 0.0
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    print(f"\n🌟 Epoch {epoch}")
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_iou = val_one_epoch(model, val_loader, loss_fn, device)

    print(f"📉 Train Loss: {train_loss:.4f} | 📊 Val Loss: {val_loss:.4f} | 🟩 Val IoU: {val_iou:.4f}")

    # ✅ Save model based on lowest train loss
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "best_by_loss.pth")
        print("💾 Saved best_by_loss.pth ✅")

    # ✅ Save model based on highest validation IoU
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_by_iou.pth")
        print("💾 Saved best_by_iou.pth ✅")

print("\n✅ Training complete.")
