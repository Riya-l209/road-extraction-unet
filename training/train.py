# # training/train.py
# print("🚀 Script started")

# import os
# import torch
# from torch import nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from model import UNet
# from losses import CombinedLoss
# from dataset import RoadDataset


# def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
#     model.train()
#     total_loss = 0
#     for images, masks in tqdm(dataloader, desc="Training"):
#         images, masks = images.to(device), masks.to(device)
#         preds = model(images)
#         loss = loss_fn(preds, masks)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(dataloader)

# def calculate_iou(preds, masks):
#     preds = (preds.sigmoid() > 0.5).float()
#     intersection = (preds * masks).sum((1, 2, 3))
#     union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
#     return (intersection / (union + 1e-6)).mean().item()

# def val_one_epoch(model, dataloader, loss_fn, device, save_output=False):
#     model.eval()
#     total_loss, total_iou = 0, 0

#     os.makedirs("outputs", exist_ok=True)

#     with torch.no_grad():
#         for idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validation")):
#             images, masks = images.to(device), masks.to(device)
#             preds = model(images)
#             loss = loss_fn(preds, masks)
#             total_loss += loss.item()
#             total_iou += calculate_iou(preds, masks)

#             # Save output masks for first few images
#             if save_output and idx < 2:
#                 import torchvision
#                 pred_mask = (preds.sigmoid() > 0.5).float()
#                 torchvision.utils.save_image(pred_mask, f"outputs/pred_{idx}.png")
#                 torchvision.utils.save_image(images, f"outputs/input_{idx}.png")
#                 torchvision.utils.save_image(masks, f"outputs/gt_{idx}.png")

#     return total_loss / len(dataloader), total_iou / len(dataloader)


# if __name__ == "__main__":
#     print("🔧 Preparing data and model...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     train_ds = RoadDataset(
#     image_dir=r"C:\\Users\\harsh\Downloads\\road-extraction-unet-vinayak-datamanage\data\\raw\\preview\\images",
#     mask_dir=r"C:\\Users\\harsh\Downloads\\road-extraction-unet-vinayak-datamanage\data\\raw\\preview\\masks",
#     transform=transform)

#     val_ds = RoadDataset("dataset/valid/images", "dataset/valid/masks", transform)

#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=4)

#     model = UNet().to(device)
#     loss_fn = CombinedLoss(alpha=0.3)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     best_iou = 0
#     num_epochs = 10

#     for epoch in range(1, num_epochs + 1):
#         print(f"\n🌟 Epoch {epoch}")
#         train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
#         val_loss, val_iou = val_one_epoch(model, val_loader, loss_fn, device, save_output=(epoch == num_epochs))

#         print(f"📉 Train Loss: {train_loss:.4f} | 📊 Val Loss: {val_loss:.4f} | 🟩 Val IoU: {val_iou:.4f}")

#         if val_iou > best_iou:
#             best_iou = val_iou
#             torch.save(model.state_dict(), "best_model.pth")
#             print("💾 Best model saved")

#     print("\n✅ Training complete.")



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
