print("🚀 Starting train_dummydata.py")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Training function ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# --- Validation function ---
def val_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# --- Main block ---
if __name__ == "__main__":
    print("✅ Inside main block")

    from model import UNet  # Make sure model.py exists and defines UNet

    # Dummy dataset: 8 random RGB images and binary masks
    dummy_images = torch.randn(8, 3, 256, 256)
    dummy_masks = torch.randint(0, 2, (8, 1, 256, 256)).float()
    dataset = TensorDataset(dummy_images, dummy_masks)
    loader = DataLoader(dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model = UNet().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("🏁 Running training on dummy data...")
    train_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
    print(f"✅ Train Loss: {train_loss:.4f}")

    print("🔍 Running validation on dummy data...")
    val_loss = val_one_epoch(model, loader, loss_fn, device)
    print(f"✅ Validation Loss: {val_loss:.4f}")
