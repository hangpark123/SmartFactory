import os
import cv2
import pandas as pd
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate, RandomScale, BboxParams

# 경로 설정
image_dir = r"C:\Users\edu31\YoloV8\captured_images\camera_1"  # 원본 이미지 경로
csv_file = r"C:\Users\edu31\YoloV8\labels\camera_1.csv"  # 라벨링 CSV 파일 경로
output_image_dir = r"C:\Users\edu31\YoloV8\dataset_camera1\images\train"  # 증강된 이미지 저장 경로
output_label_dir = r"C:\Users\edu31\YoloV8\dataset_camera1\labels\train"  # 증강된 라벨 저장 경로

# 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 증강 횟수 설정
augmentation_count = 999

# 증강 파이프라인 설정
transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.2),
    Rotate(limit=30, p=0.5),
    RandomScale(scale_limit=0.2, p=0.5)
], bbox_params=BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

# CSV 파일 불러오기
df = pd.read_csv(csv_file)

# 이미지와 라벨 증강
for _, row in df.iterrows():
    image_path = os.path.join(image_dir, row['이미지'])
    if not os.path.exists(image_path):
        print(f"이미지 {row['이미지']}를 찾을 수 없습니다.")
        continue

    # 이미지 불러오기
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    bboxes = [[row['X1'], row['Y1'], row['X2'], row['Y2']]]  # Pascal VOC 좌표 형식
    labels = [0]  # YOLO 형식 라벨 인덱스

    # 증강 수행
    for i in range(augmentation_count):
        try:
            # 증강 이미지 생성
            augmented = transform(image=image, bboxes=bboxes, labels=labels)

            # 고유한 이름 생성
            base_image_name = os.path.splitext(row['이미지'])[0]
            output_image_name = f"{base_image_name}_aug_{i+1:03}.jpg"
            output_image_path = os.path.join(output_image_dir, output_image_name)

            # 증강된 이미지 저장
            cv2.imwrite(output_image_path, augmented['image'])

            # 증강된 바운딩 박스 TXT 파일로 저장
            output_label_file = os.path.join(output_label_dir, f"{os.path.splitext(output_image_name)[0]}.txt")
            with open(output_label_file, "w") as f:
                for bbox, label in zip(augmented['bboxes'], augmented['labels']):
                    # YOLO 형식으로 바운딩 박스 변환
                    x_center = (bbox[0] + bbox[2]) / 2 / width
                    y_center = (bbox[1] + bbox[3]) / 2 / height
                    box_width = (bbox[2] - bbox[0]) / width
                    box_height = (bbox[3] - bbox[1]) / height

                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

            print(f"증강된 데이터 저장 완료: {output_image_name}, 라벨: {output_label_file}")

        except Exception as e:
            print(f"증강 중 오류 발생: {e}")

print("데이터 증강 및 라벨 저장 완료!")