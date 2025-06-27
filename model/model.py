import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.output(dec1))

class SatelliteRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.png'))
            mask = Image.open(mask_path).convert('L')
        else:
            mask = torch.zeros(image.size[1], image.size[0])  
            
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def process_images(model, image_dir, output_dir, batch_size=4, device='cuda'):
  
    os.makedirs(output_dir, exist_ok=True)
    
  
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])
    
    
    dataset = SatelliteRoadDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
   
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
        
            outputs = model(images)
            
        
            for i in range(outputs.shape[0]):
                img_idx = batch_idx * batch_size + i
                if img_idx >= len(dataset):
                    break  
                
                
                output = outputs[i].cpu().squeeze().numpy()
                output = (output * 255).astype(np.uint8)  
                
                
                output_img = Image.fromarray(output)
                output_path = os.path.join(output_dir, f'road_extraction_{dataset.images[img_idx]}')
                output_img.save(output_path)
                
                print(f'Saved: {output_path}')

if __name__ == '__main__':
    
    model = UNet()
    
   
    image_dir = 'path/to/your/40/images'  
    output_dir = 'path/to/output/directory' 
    
   
    process_images(model, image_dir, output_dir, batch_size=4)
