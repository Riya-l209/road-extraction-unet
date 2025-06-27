import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import UNet

# Paths
image_dir = r"C:\Users\BIT\Desktop\new road\images"
mask_dir = r"C:\Users\BIT\Desktop\new road\masks"
best_model_path = "best_model.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = UNet().to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Get list of test images
images = sorted(os.listdir(image_dir))

# Visualize predictions
for img_name in images:  # Just show 5 predictions
    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(model(input_tensor))

        # Debug: Check min and max values
        print(f"Image: {img_name} | Min: {pred.min().item():.4f}, Max: {pred.max().item():.4f}")

        # Apply threshold (try 0.3 if 0.5 is too strict)
        pred = (pred > 0.5).float()

    # Convert tensors to image format
    pred_mask = pred.squeeze().cpu().numpy()
    gt_mask = transform(mask).squeeze().cpu().numpy()
    image_np = transform(image).permute(1, 2, 0).cpu().numpy()

    # Plot side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title("Predicted Mask")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
