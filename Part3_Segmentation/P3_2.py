import os
import shutil
import json
import numpy as np
from PIL import Image
import nibabel as nib

# PNG -> NIfTI 변환 함수
def convert_png_to_nii(png_path, nii_path):
    img = Image.open(png_path)
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]
    affine = np.eye(4)  
    nii_img = nib.Nifti1Image(img_array, affine)
    nib.save(nii_img, nii_path)

# 데이터셋 기본 정보
dataset_id = 542
dataset_name = "COVID19"
nnunet_raw_path = f"/home/suyeon/nnUNet/raw/Dataset{dataset_id}_{dataset_name}"

# 원본 데이터 경로
source_dir = "/home/suyeon/nnUNet/raw/COVID19_dataset"

# nnU-Net 형식에 맞게 폴더 생성
os.makedirs(os.path.join(nnunet_raw_path, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(nnunet_raw_path, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(nnunet_raw_path, "imagesTs"), exist_ok=True)

# 학습 이미지 및 라벨 이동 (PNG로 임시 저장)
train_images = os.listdir(os.path.join(source_dir, "train", "images"))

for img_file in train_images:
    base_name = os.path.splitext(img_file)[0]

    # 이미지 이동 (파일명 뒤에 _0000 추가)
    shutil.move(
        os.path.join(source_dir, "train", "images", img_file),
        os.path.join(nnunet_raw_path, "imagesTr", f"{base_name}_0000.png")
    )

    # 마스크 이동
    shutil.move(
        os.path.join(source_dir, "train", "masks", img_file),
        os.path.join(nnunet_raw_path, "labelsTr", f"{base_name}.png")
    )

# 테스트 이미지 이동
test_images = os.listdir(os.path.join(source_dir, "test", "images"))

for img_file in test_images:
    base_name = os.path.splitext(img_file)[0]
    
    shutil.move(
        os.path.join(source_dir, "test", "images", img_file),
        os.path.join(nnunet_raw_path, "imagesTs", f"{base_name}_0000.png")
    )

# Step 2: 마스크 값 변환 (255 → 1)
label_dir = os.path.join(nnunet_raw_path, "labelsTr")

for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    mask = Image.open(label_path).convert("L")  
    mask_array = np.array(mask)
    mask_array[mask_array == 255] = 1  
    new_mask = Image.fromarray(mask_array)
    new_mask.save(label_path)

print("마스크 값 변환 완료: 255 → 1")

# Step 3: 마스크 크기 조정 (이미지와 동일한 크기로 맞춤)
image_dir = os.path.join(nnunet_raw_path, "imagesTr")

for image_file in os.listdir(image_dir):
    if not image_file.endswith(".png"):
        continue  
    
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace("_0000", ""))
    
    if not os.path.exists(label_path):
        print(f"라벨 파일 없음: {label_path}")
        continue

    image = Image.open(image_path)
    target_size = image.size  

    mask = Image.open(label_path).convert("L")
    resized_mask = mask.resize(target_size, Image.NEAREST)  
    resized_mask.save(label_path)

print("모든 마스크 크기 변환 완료!")

# Step 4: PNG 파일을 NIfTI(.nii.gz)로 변환
# imagesTr 변환
for file in os.listdir(image_dir):
    if file.endswith(".png"):
        png_path = os.path.join(image_dir, file)
        nii_filename = file.replace(".png", ".nii.gz")
        nii_path = os.path.join(image_dir, nii_filename)
        convert_png_to_nii(png_path, nii_path)
        os.remove(png_path)  # PNG 삭제

# labelsTr 변환
for file in os.listdir(label_dir):
    if file.endswith(".png"):
        png_path = os.path.join(label_dir, file)
        nii_filename = file.replace(".png", ".nii.gz")
        nii_path = os.path.join(label_dir, nii_filename)
        convert_png_to_nii(png_path, nii_path)
        os.remove(png_path)

# imagesTs 변환
test_image_dir = os.path.join(nnunet_raw_path, "imagesTs")
for file in os.listdir(test_image_dir):
    if file.endswith(".png"):
        png_path = os.path.join(test_image_dir, file)
        nii_filename = file.replace(".png", ".nii.gz")
        nii_path = os.path.join(test_image_dir, nii_filename)
        convert_png_to_nii(png_path, nii_path)
        os.remove(png_path)

print("PNG 파일을 NIfTI 형식으로 변환 완료!")

# Step 5: dataset.json 파일 생성
nii_image_dir = os.path.join(nnunet_raw_path, "imagesTr")
dataset_info = {
    "dataset_id": dataset_id,  
    "name": "COVID19",
    "description": "Lung mask segmentation dataset for COVID-19",
    "tensorImageSize": "2D",
    "reference": "https://www.kaggle.com/",
    "licence": "CC BY 4.0",
    "release": "1.0",
    "channel_names": {"0": "X-ray"},  
    "labels": {"background": 0, "lung": 1},  
    "numTraining": len(os.listdir(nii_image_dir)),  
    "numTest": len(os.listdir(os.path.join(nnunet_raw_path, "imagesTs"))),  
    "file_ending": ".nii.gz",  
    "training": [
        {
            "image": f"./imagesTr/{file}",
            "label": f"./labelsTr/{file.replace('_0000.nii.gz', '.nii.gz')}"
        }
        for file in sorted(os.listdir(nii_image_dir))
    ],
    "test": [
        f"./imagesTs/{file}"
        for file in sorted(os.listdir(os.path.join(nnunet_raw_path, "imagesTs")))
    ]
}

# JSON 저장
json_path = os.path.join(nnunet_raw_path, "dataset.json")
with open(json_path, "w") as json_file:
    json.dump(dataset_info, json_file, indent=4)

print(f"JSON 파일 생성 완료: {json_path}")
print("모든 작업 완료: nnU-Net 형식으로 데이터 정리 및 NIfTI 변환 완료!")


# nnUNetv2_plan_and_preprocess -d 542 --verify_dataset_integrity
# 제일 기본으로 학습 시작
# nnUNetv2_train Dataset542_COVID19 2d 0 
# nnUNetv2_predict -i /home/suyeon/nnUNet/raw/Dataset542_COVID19/imagesTs -o /home/suyeon/nnUNet/predictions/Dataset542_COVID19 -d Dataset542_COVID19 -c 2d -f 0