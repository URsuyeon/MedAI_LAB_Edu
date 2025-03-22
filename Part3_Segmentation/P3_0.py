import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 디렉토리 설정
image_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/images"
mask_dir = "/mnt/c/Users/SU/Desktop/materials/COVID-19_Radiography_Dataset/Normal/masks"
prediction_dir = "/home/suyeon/nnUNet/results/Dataset546_COVID19/visual/masks"

# 샘플 번호
sample_idx = "7622"  # 문자열로 변환하여 포함 여부 확인

# 특정 번호가 포함된 파일 찾기
def find_file(directory, sample_idx):
    matching_files = [f for f in os.listdir(directory) if sample_idx in f and f.endswith('.png')]
    return os.path.join(directory, matching_files[0]) if matching_files else None

# 파일 경로 설정
image_path = find_file(image_dir, sample_idx)
mask_path = find_file(mask_dir, sample_idx)
prediction_path = find_file(prediction_dir, sample_idx)

# 파일 존재 여부 확인
if not image_path:
    raise FileNotFoundError(f"No image file found containing: {sample_idx}")
if not mask_path:
    raise FileNotFoundError(f"No mask file found containing: {sample_idx}")
if not prediction_path:
    raise FileNotFoundError(f"No prediction file found containing: {sample_idx}")

# 이미지 로드 및 변환
image = Image.open(image_path).convert("L").resize((256, 256))  
mask = Image.open(mask_path).convert("L").resize((256, 256))  
prediction = Image.open(prediction_path).convert("L").resize((256, 256))

# NumPy 배열 변환
image_np = np.array(image)
mask_np = np.array(mask)
prediction_np = np.array(prediction)

# 차이 계산 (절대값)
diff = np.abs(mask_np - prediction_np)

# 플롯 생성
plt.figure(figsize=(20, 7))

# (1) 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Original")
plt.axis("off")

# (2) 원본 + Ground Truth Mask + Prediction Mask
plt.subplot(1, 3, 2)
plt.imshow(image_np, cmap='gray')
plt.imshow(mask_np, cmap='jet', alpha=0.5)  # Ground Truth Mask
plt.imshow(prediction_np, cmap='hot', alpha=0.3)  # Prediction Mask
plt.title("Original + Mask + Prediction")
plt.axis("off")

# (4) 차이 이미지
plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='coolwarm')
plt.title("Difference (Mask vs Prediction)")
plt.axis("off")

# 표시
plt.show()
