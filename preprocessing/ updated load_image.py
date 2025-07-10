import cv2
import os
import matplotlib.pyplot as plt
image_dir = r"C:\Users\BIT\OneDrive\Desktop\road-extraction-unet-main\data\raw\preview\images"
mask_dir = r"C:\Users\BIT\OneDrive\Desktop\road-extraction-unet-main\data\raw\preview\masks"

file_names = [f for f in os.listdir(image_dir) if f.endswith(".png")]
for file_name in file_names:
    image_path = os.path.join(image_dir, file_name)
    mask_path = os.path.join(mask_dir, file_name)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        continue
    image = cv2.imread(image_path)
    image = image / 255.0 if image is not None else None
     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is not None and mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Image: {file_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: {file_name}")
        plt.axis('off')

        plt.show()
    else:
        print(f"⚠ Failed to load image or mask: {file_name}")
