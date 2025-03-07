# 0. ants.dicom_read() 함수 이용해서 아래 폴더 읽기
# /public/study/materials/3d/dicom/S.D PRE-CONTRAST
#   Image.spacing, Image.shape, Image.dtype, Image.dimension 확인하여 기본 특성 확인
#   Image.numpy() method 이용하여 numpy 배열 얻어보기 + 시각화 해보기
#   Image.plot(axis=N) 이용하여 시각화해보고 위랑 비교
#   ants.image_read() 함수로 nifti 실습파일 아무거나 읽고 위 과정 반복

import ants
import matplotlib.pyplot as plt
import numpy as np

dicom_path = "/mnt/c/Users/SU/Desktop/materials/3d/dicom/S.D PRE-CONTRAST"
dicom_image = ants.dicom_read(dicom_path)

# 기본 속성 확인
print("Spacing:", dicom_image.spacing)
print("Shape:", dicom_image.shape)
print("Data Type:", dicom_image.dtype)
print("Dimension:", dicom_image.dimension)

# NumPy 배열 변환
numpy_array = dicom_image.numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# NumPy 변환 후 시각화 
axes[0].imshow(numpy_array[:, :, numpy_array.shape[2] // 2], cmap="gray")  # Axial (XY)
axes[0].set_title("Axial")

axes[1].imshow(numpy_array[:, numpy_array.shape[1] // 2, :], cmap="gray")  # Coronal (XZ)
axes[1].set_title("Coronal")

axes[2].imshow(numpy_array[numpy_array.shape[0] // 2, :, :], cmap="gray")  # Sagittal (YZ)
axes[2].set_title("Sagittal")

plt.tight_layout()
plt.show()
