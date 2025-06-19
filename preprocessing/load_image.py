import cv2
import os
import matplotlib.pyplot as plt
image_path = r"C:\Users\BIT\OneDrive\Desktop\road-extraction-unet-main\data\raw\preview\images\10078660_15.png"
mask_path = r"C:\Users\BIT\OneDrive\Desktop\road-extraction-unet-main\data\raw\preview\masks\10078660_15.png"
if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
else:
    print(f"Image file found: {image_path}")

if not os.path.exists(mask_path):
    print(f" Mask file not found: {mask_path}")
else:
    print(f" Mask file found: {mask_path}")
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if image is not None:
    image = image / 255.0
if image is not None and mask is not None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Satellite Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask Image")
    plt.axis('off')
    plt.show()
else:
    print("Image or mask not loaded. Please check the file paths and file integrity.")