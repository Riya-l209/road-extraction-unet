from dataset import RoadDataset
from torchvision import transforms
import traceback

# Define your paths
image_dir = r"C:\Users\harsh\Music\OneDrive\Desktop\road-extraction-unet\all new files\images"
mask_dir = r"C:\Users\harsh\Music\OneDrive\Desktop\road-extraction-unet\all new files\masks"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create the dataset
try:
    ds = RoadDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    print(f"✅ Loaded dataset with {len(ds)} samples")
    
    # Try loading first sample
    img, mask = ds[0]
    print(f"🖼️ Image shape: {img.shape}")
    print(f"🛣️ Mask shape: {mask.shape}")

except FileNotFoundError as e:
    print("❌ File not found error:")
    print(e)

except Exception as e:
    print("❌ Other error occurred:")
    traceback.print_exc()
