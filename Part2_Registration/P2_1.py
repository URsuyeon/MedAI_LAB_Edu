# 1. ants.resample_image() 활용
#   위에서 읽은 파일을 이용해 1x1x1mm resolution을 가지도록 resampling 해보기 (interpolation bspline 적용) + 결과 시각화
#   위에서 읽은 파일을 이용해 128x128x128 resolution을 가지도록 resampling 해보기 (interpolation bspline 적용) + 결과 시각화

import ants
import matplotlib.pyplot as plt

# DICOM 이미지 읽기
dicom_path = "/mnt/c/Users/SU/Desktop/materials/3d/dicom/S.D PRE-CONTRAST"
image = ants.dicom_read(dicom_path)

# 원본 이미지 정보
print("Original image info:")
print(image)

# 1x1x1 해상도로 리샘플링 (B-Spline 보간법 사용)
resampled_image_1 = ants.resample_image(image, (1.0, 1.0, 1.0), use_voxels=False, interp_type=4) # 4 (bspline)

# 128x128x128 해상도로 리샘플링 (B-Spline 보간법 사용)
resampled_image_128 = ants.resample_image(image, (128, 128, 128), use_voxels=True, interp_type=4) # 4 (bspline)

# 시각화 
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].imshow(image.numpy()[:,:,0], cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(resampled_image_1.numpy()[:,:,0], cmap='gray')
axes[1].set_title("Resampled Image (1x1x1)")
axes[1].axis('off')

axes[2].imshow(resampled_image_128.numpy()[:,:,0], cmap='gray')
axes[2].set_title("Resampled Image (128x128x128)")
axes[2].axis('off')

plt.suptitle("Step 1: Resampling", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()