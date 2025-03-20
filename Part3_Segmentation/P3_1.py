import os
import shutil
import random

image_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/images"
mask_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/masks"

output_dir = "/home/suyeon/nnUNet/raw/COVID19_dataset"
os.makedirs(output_dir, exist_ok=True)

for split in ["train", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

data = list(zip(image_files, mask_files))
random.shuffle(data)

train_ratio = 0.8
train_split = int(len(data) * train_ratio)

train_data = data[:train_split]
test_data = data[train_split:]

def copy_files(data_list, split):
    for img_file, mask_file in data_list:
        shutil.copy(os.path.join(image_dir, img_file), os.path.join(output_dir, split, "images", img_file))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(output_dir, split, "masks", mask_file))

copy_files(train_data, "train")
copy_files(test_data, "test")

print("데이터셋 분할 및 복사 완료!")