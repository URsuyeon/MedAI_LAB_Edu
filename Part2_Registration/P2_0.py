# 0. ants.dicom_read() 함수 이용해서 아래 폴더 읽기
# /public/study/materials/3d/dicom/S.D PRE-CONTRAST
#   Image.spacing, Image.shape, Image.dtype, Image.dimension 확인하여 기본 특성 확인
#   Image.numpy() method 이용하여 numpy 배열 얻어보기 + 시각화 해보기
#   Image.plot(axis=N) 이용하여 시각화해보고 위랑 비교
#   ants.image_read() 함수로 nifti 실습파일 아무거나 읽고 위 과정 반복