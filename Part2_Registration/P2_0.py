# 0. ants.dicom_read() 함수 이용해서 아래 폴더 읽기
# /public/study/materials/3d/dicom/S.D PRE-CONTRAST
#   Image.spacing, Image.shape, Image.dtype, Image.dimension 확인하여 기본 특성 확인
#   Image.numpy() method 이용하여 numpy 배열 얻어보기 + 시각화 해보기
#   Image.plot(axis=N) 이용하여 시각화해보고 위랑 비교
#   ants.image_read() 함수로 nifti 실습파일 아무거나 읽고 위 과정 반복

import ants
import matplotlib.pyplot as plt
import numpy as np

def visual(image_path, is_dicom = False):
    
    image = ants.dicom_read(image_path) if is_dicom else ants.image_read(image_path)

    # 기본 속성 확인
    print("Spacing:", image.spacing)
    print("Shape:", image.shape)
    print("Data Type:", image.dtype)
    print("Dimension:", image.dimension)

    # NumPy 배열 변환
    numpy_array = image.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # NumPy 변환 후 시각화 
    axes[0, 0].imshow(numpy_array[:, :, numpy_array.shape[2] // 2], cmap="gray")  # Axial (XY)
    axes[0, 0].set_title("NumPy - Axial")

    axes[0, 1].imshow(numpy_array[:, numpy_array.shape[1] // 2, :], cmap="gray")  # Coronal (XZ)
    axes[0, 1].set_title("NumPy - Coronal")

    axes[0, 2].imshow(numpy_array[numpy_array.shape[0] // 2, :, :], cmap="gray")  # Sagittal (YZ)
    axes[0, 2].set_title("NumPy - Sagittal")

    # ANTs plot 시각화 
    for axis in range(3):
        slice_indices = np.any(numpy_array, axis=(axis + 1) % 3)  
        valid_slices = np.where(slice_indices)[0]  
        
        slice_index = min(valid_slices[len(valid_slices) // 2], numpy_array.shape[axis] - 1)

        transformed_img = ants.slice_image(image, axis=axis, idx=slice_index).numpy()
        axes[1, 2-axis].imshow(transformed_img, cmap="gray")
        axes[1, 2-axis].set_title(f"ANTs - Axis {axis}")

    plt.suptitle(f"{'DICOM' if is_dicom else 'NIfTI'} NumPy vs ANTs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

dicom_path = "/mnt/c/Users/SU/Desktop/materials/3d/dicom/S.D PRE-CONTRAST"
nifti_path = "/mnt/c/Users/SU/Desktop/materials/3d/nifti/TCGA-02-0006_1996.08.23_flair.nii.gz"

visual(dicom_path, is_dicom=True)
visual(nifti_path, is_dicom=False)