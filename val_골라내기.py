import os
import shutil
import random

# 경로 설정
train_image_dir = r"C:\Users\edu31\YoloV8\dataset_camera2\images\train"
val_image_dir = r"C:\Users\edu31\YoloV8\dataset_camera2\images\val"
train_label_dir = r"C:\Users\edu31\YoloV8\dataset_camera2\labels\train"
val_label_dir = r"C:\Users\edu31\YoloV8\dataset_camera2\labels\val"

# 검증 비율 설정
val_split_ratio = 0.2

# 폴더 생성
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 트레인 디렉토리의 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# 검증 이미지 개수 계산
val_count = int(len(image_files) * val_split_ratio)

# 검증 이미지 랜덤 선택
val_images = random.sample(image_files, val_count)

# 검증 이미지 및 라벨 이동
for img_file in val_images:
    # 이미지 이동
    src_image_path = os.path.join(train_image_dir, img_file)
    dst_image_path = os.path.join(val_image_dir, img_file)
    shutil.move(src_image_path, dst_image_path)

    # 라벨 이동
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label_path = os.path.join(train_label_dir, label_file)
    dst_label_path = os.path.join(val_label_dir, label_file)

    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
        print(f"이미지 및 라벨 이동 완료: {img_file}, {label_file}")
    else:
        print(f"라벨 파일이 없습니다: {label_file}")

print(f"검증 이미지 및 라벨 {val_count}장 이동 완료!")
