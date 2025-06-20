print("🚀 Script started")

import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet
from losses import CombinedLoss
from dataset import RoadDataset

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    print("🔧 Preparing data and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = RoadDataset(
        image_dir=r"C:\\Users\\harsh\\Music\\OneDrive\Desktop\\road-extraction-unet\\all new files\\images",
        mask_dir=r"C:\\Users\\harsh\\Music\\OneDrive\Desktop\\road-extraction-unet\\all new files\\masks",
        transform=transform)
    

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = UNet().to(device)
    loss_fn = CombinedLoss(alpha=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        print(f"\n🌟 Epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"📉 Train Loss: {train_loss:.4f}")

    print("\n✅ Training complete.")
