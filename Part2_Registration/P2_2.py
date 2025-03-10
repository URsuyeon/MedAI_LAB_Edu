# 2. ants.registration() 활용

#   Registration 수행 및 전후 비교 (intra-patient registration)
#       File: /public/study/materials/reg/GBM 안에 있는 파일들 활용
#       FLAIR → T1c (moving → fixed) 수행
#           Tranformation: SyN
#           outprefix: 반드시 경로로 지정할 것 (ex. /F/myfolder/myprefix)
#           출력되는 3개 파일들이 무슨 정보를 담고있는지 알아보기
#               [prefix0GenericAffine.mat,
#               prefix1Warp.nii.gz,
#               prefix1InverseWarp.nii.gz)
#           Registration 결과물 파일로 저장하기 (Hint. ant.image_write() 사용)
#           Registration 전후 시각화

#       T2 → FLAIR (moving → fixed) 아래 옵션 사용하여 수행, 생성 파일 비교
#           Option1:
#               Transformation: Affine
#               outprefix: 반드시 경로로 지정할 것 (ex. /F/myfolder/myprefix)
#           Option2:
#               Transformation: SyN
#               outprefix: 반드시 경로로 지정할 것 (ex. /F/myfolder/myprefix)

import ants
import numpy as np
import matplotlib.pyplot as plt
import os

def create_output_directory(path):
    os.makedirs(path, exist_ok=True)

def load_images(paths):
    return {name: ants.image_read(path) for name, path in paths.items()}

def register_images(fixed, moving, transform_type, outprefix):
    moving_resampled = ants.resample_image(moving, fixed.shape, use_voxels=True)
    reg_result = ants.registration(fixed, moving_resampled, type_of_transform=transform_type, outprefix=outprefix)
    
    return reg_result

def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min) if img_max > img_min else img

def visual(fixed, moving, warped, title, output_dir, rows=4, cols=3):
    
    moving_resampled = ants.resample_image(moving, fixed.shape, use_voxels=True)
    warped_resampled = ants.resample_image(warped, fixed.shape, use_voxels=True)
    
    z_dim = min(fixed.shape[2], moving_resampled.shape[2], warped_resampled.shape[2])
    slice_indices = np.linspace(z_dim//8, z_dim*7//8, rows*cols, dtype=int)
    
    fig, axes = plt.subplots(rows, cols*2, figsize=(7,4))
    plt.suptitle(title, fontsize=16)
    
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    
    for i, idx in enumerate(slice_indices):
        row, col = divmod(i, cols)
        fixed_slice = normalize(fixed.numpy()[:, :, idx])
        moving_slice = normalize(moving_resampled.numpy()[:, :, idx])
        warped_slice = normalize(warped_resampled.numpy()[:, :, idx])
        
        axes[row, col].imshow(fixed_slice, cmap='gray')
        axes[row, col].imshow(moving_slice, cmap='hot', alpha=0.5)
        axes[row, col].axis('off')
        
        axes[row, col+cols].imshow(fixed_slice, cmap='gray')
        axes[row, col+cols].imshow(warped_slice, cmap='hot', alpha=0.5)
        axes[row, col+cols].axis('off')
    
    plt.savefig(f"{output_dir}/{title.replace(' ', '_').lower()}_grid.png")

output_dir = "/home/suyeon/result"
create_output_directory(output_dir)

image_paths = {
    "FLAIR": "/mnt/c/Users/SU/Desktop/materials/reg/GBM/FLAIR.nii.gz",
    "T1c": "/mnt/c/Users/SU/Desktop/materials/reg/GBM/T1c.nii.gz",
    "T2": "/mnt/c/Users/SU/Desktop/materials/reg/GBM/T2.nii.gz"
}

images = load_images(image_paths)

# 정합, 결과 저장
registrations = {
    "FLAIR_to_T1c_Affine": register_images(images["T1c"], images["FLAIR"], "Affine", f"{output_dir}/flair_to_t1c_affine_"),
    "FLAIR_to_T1c_SyN": register_images(images["T1c"], images["FLAIR"], "SyN", f"{output_dir}/flair_to_t1c_syn_"),
    "T2_to_FLAIR_Affine": register_images(images["FLAIR"], images["T2"], "Affine", f"{output_dir}/t2_to_flair_affine_"),
    "T2_to_FLAIR_SyN": register_images(images["FLAIR"], images["T2"], "SyN", f"{output_dir}/t2_to_flair_syn_")
}

# 결과 시각화
visual(images["T1c"], images["FLAIR"], registrations["FLAIR_to_T1c_Affine"]['warpedmovout'], "FLAIR to T1c (Affine)", output_dir)
visual(images["T1c"], images["FLAIR"], registrations["FLAIR_to_T1c_SyN"]['warpedmovout'], "FLAIR to T1c (SyN)", output_dir)
visual(images["FLAIR"], images["T2"], registrations["T2_to_FLAIR_Affine"]['warpedmovout'], "T2 to FLAIR (Affine)", output_dir)
visual(images["FLAIR"], images["T2"], registrations["T2_to_FLAIR_SyN"]['warpedmovout'], "T2 to FLAIR (SyN)", output_dir)