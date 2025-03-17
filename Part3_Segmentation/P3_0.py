import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/images"
mask_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/masks"

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

sample_idx = 200  
image_path = os.path.join(image_dir, image_files[sample_idx])
mask_path = os.path.join(mask_dir, mask_files[sample_idx])

image = Image.open(image_path).convert("L").resize((256, 256))  
mask = Image.open(mask_path).convert("L").resize((256, 256)) 

image_np = np.array(image)
mask_np = np.array(mask)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Lung CT Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.imshow(mask, cmap='jet', alpha=0.5)
plt.title("Lung Mask Overlay")
plt.axis("off")

plt.show()