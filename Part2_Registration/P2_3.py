# 3. ants.apply_transform() 활용
#   SyN transformation file들 이용해서 T2영상
#   T1c 공간에 옮겨보고 시각화 (Hint. Tranformation 순서 affine → deformation, Two-step transformation)

import ants
import numpy as np
import matplotlib.pyplot as plt 
import os

def create_output_directory(path):
    os.makedirs(path, exist_ok=True)
    
def load_images(paths):
    return {name: ants.image_read(path) for name, path in paths.items()}

def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min) if img_max > img_min else img

def visual(fixed, moving, warped, title, output_dir, rows=4, cols=3):
    moving_resampled = ants.resample_image(moving, fixed.shape, use_voxels=True)
    warped_resampled   = ants.resample_image(warped, fixed.shape, use_voxels=True)
    
    z_dim = min(fixed.shape[2], moving_resampled.shape[2])
    slice_indices = np.linspace(z_dim//8, z_dim*7//8, rows*cols, dtype=int)
    
    fig, axes = plt.subplots(rows, cols*2, figsize=(7,4))
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    
    for i, idx in enumerate(slice_indices):
        row, col = divmod(i, cols)
        fixed_slice = normalize(fixed.numpy()[:, :, idx])
        moving_slice = normalize(moving_resampled.numpy()[:, :, idx])
        warped_slice  = normalize(warped_resampled.numpy()[:, :, idx])
        
        axes[row, col].imshow(fixed_slice, cmap='gray')
        axes[row, col].imshow(moving_slice, cmap='hot', alpha=0.5)
        axes[row, col].axis('off')
        
        axes[row, col+cols].imshow(fixed_slice, cmap='gray')
        axes[row, col+cols].imshow(warped_slice, cmap='hot', alpha=0.5)
        axes[row, col+cols].axis('off')
    
    plt.savefig(f"{output_dir}/{title.replace(' ', '_').lower()}_grid.png")

output_dir = "/home/suyeon/result_apply"
create_output_directory(output_dir)

image_paths = {
    "T1c": "/mnt/c/Users/SU/Desktop/materials/reg/GBM/T1c.nii.gz",
    "T2": "/mnt/c/Users/SU/Desktop/materials/reg/GBM/T2.nii.gz"
}

images = load_images(image_paths)

registration_dir = "/home/suyeon/result"

flair_to_t1c_affine = f"{registration_dir}/flair_to_t1c_affine_0GenericAffine.mat"
flair_to_t1c_warp   = f"{registration_dir}/flair_to_t1c_syn_1Warp.nii.gz"

t2_to_flair_affine  = f"{registration_dir}/t2_to_flair_affine_0GenericAffine.mat"
t2_to_flair_warp    = f"{registration_dir}/t2_to_flair_syn_1Warp.nii.gz"

transform_list = [flair_to_t1c_warp, flair_to_t1c_affine, t2_to_flair_warp, t2_to_flair_affine]

transformed_t2 = ants.apply_transforms(fixed=images["T1c"], moving=images["T2"], transformlist=transform_list)

ants.image_write(transformed_t2, f"{output_dir}/t2_in_t1c_space.nii.gz")

visual(images["T1c"],  images["T2"], transformed_t2, "T2 to T1c", output_dir)