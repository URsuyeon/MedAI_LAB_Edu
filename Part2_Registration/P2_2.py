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
