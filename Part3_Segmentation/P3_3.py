import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import nibabel as nib  

# 데이터셋 기본 정보
dataset_id = 546
dataset_name = "COVID19"
Dataset = f"Dataset{dataset_id}_{dataset_name}"

# 경로 설정
original_images_dir = f"/home/suyeon/nnUNet/raw/{Dataset}/imagesTs"
mask_images_dir = f"/home/suyeon/nnUNet/results/{Dataset}/predictions_2d"
output_dir = f"/home/suyeon/nnUNet/results/{Dataset}/visual"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

# NIfTI 파일 목록 가져오기
original_files = glob.glob(os.path.join(original_images_dir, "*.nii.gz"))
mask_files = glob.glob(os.path.join(mask_images_dir, "*.nii.gz"))

if not original_files:
    print(f"원본 이미지를 찾을 수 없습니다: {original_images_dir}")
    print(f"찾은 파일: {os.listdir(original_images_dir)[:5]}... 등")
    exit(1)

if not mask_files:
    print(f"마스크 이미지를 찾을 수 없습니다: {mask_images_dir}")
    print(f"찾은 파일: {os.listdir(mask_images_dir)[:5]}... 등")
    exit(1)

print(f"원본 이미지 파일 개수: {len(original_files)}")
print(f"마스크 파일 개수: {len(mask_files)}")

# 파일 매칭 함수
def find_matching_original(mask_filename, original_files):
    mask_basename = os.path.basename(mask_filename).replace('.nii.gz', '')
    
    potential_original = os.path.join(original_images_dir, f"{mask_basename}.nii.gz")
    if os.path.exists(potential_original):
        return potential_original
    
    if "_pred" in mask_basename:
        base_without_pred = mask_basename.replace("_pred", "")
        potential_original = os.path.join(original_images_dir, f"{base_without_pred}.nii.gz")
        if os.path.exists(potential_original):
            return potential_original
    
    for orig_file in original_files:
        orig_basename = os.path.basename(orig_file)
        if mask_basename in orig_basename or orig_basename in mask_basename:
            return orig_file
    
    if "_" in mask_basename:
        base_part = mask_basename.split("_")[0]
        for orig_file in original_files:
            if base_part in os.path.basename(orig_file):
                return orig_file
    
    return None

processed_count = 0
error_count = 0

for mask_path in mask_files:
    mask_filename = os.path.basename(mask_path)
    mask_basename = mask_filename.replace('.nii.gz', '')
    
    original_path = find_matching_original(mask_path, original_files)
    
    if original_path is None:
        print(f"원본 이미지를 찾을 수 없습니다: {mask_basename}")
        error_count += 1
        continue
    
    try:
        # NIfTI 이미지 로드
        mask_nii = nib.load(mask_path)
        original_nii = nib.load(original_path)
        
        mask_data = mask_nii.get_fdata()
        original_data = original_nii.get_fdata()
        
        if len(mask_data.shape) > 3:
            print(f"다차원 데이터 감지: {mask_basename}, 모양: {mask_data.shape}")
            mask_data = mask_data[:,:,:,0]
            
        if len(original_data.shape) > 3:
            original_data = original_data[:,:,:,0]
        
        slice_idx = mask_data.shape[2] // 2
        
        if mask_data[:,:,slice_idx].max() == 0:
            for i in range(mask_data.shape[2]):
                if mask_data[:,:,i].max() > 0:
                    slice_idx = i
                    break
        
        mask_slice = mask_data[:, :, slice_idx]
        original_slice = original_data[:, :, slice_idx]
        
        if original_slice.min() != original_slice.max():
            original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min()) * 255
        original_slice = original_slice.astype(np.uint8)
        
        mask_slice = (mask_slice > 0).astype(np.uint8) * 255
        
        # 1. 마스크 시각화
        visualized_mask = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)
        visualized_mask[mask_slice > 0] = [255, 255, 255]  
        
        mask_output_path = os.path.join(output_dir, "masks", f"{mask_basename}_visualized.png")
        Image.fromarray(visualized_mask).save(mask_output_path)
        
        # 2. 원본 이미지에 마스크 오버레이
        original_rgb = np.stack([original_slice, original_slice, original_slice], axis=2)
            
        red_mask = np.zeros_like(original_rgb)
        red_mask[mask_slice > 0] = [255, 0, 0] 
        
        alpha = 0.5  
        blended = (alpha * red_mask + (1-alpha) * original_rgb).astype(np.uint8)
        
        overlay_output_path = os.path.join(output_dir, "overlays", f"{mask_basename}_overlay.png")
        Image.fromarray(blended).save(overlay_output_path)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"진행 상황: {processed_count}/{len(mask_files)} 파일 처리됨")
        
    except Exception as e:
        print(f"이미지 {mask_basename} 처리 중 오류 발생: {e}")
        error_count += 1

print(f"\n처리 완료!")
print(f"성공: {processed_count}개 파일 처리됨")
print(f"실패: {error_count}개 파일")
print(f"시각화된 마스크: {os.path.join(output_dir, 'masks')}")
print(f"오버레이 이미지: {os.path.join(output_dir, 'overlays')}")
