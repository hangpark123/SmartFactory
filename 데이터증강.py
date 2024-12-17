import os
import cv2
import pandas as pd
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate, RandomScale, BboxParams

# 경로 설정
image_dir = r"C:\Users\edu31\Desktop\YoloV8\labeled_images\labeled_images_bottom"  # 원본 이미지 경로
csv_file = r"C:\Users\edu31\Desktop\YoloV8\labeled_images\labeled_images_bottom\region_labels.csv"  # 라벨링 CSV 파일 경로
output_image_dir = r"C:\Users\edu31\Desktop\YoloV8\augmented_images\augmented_images_bottom"  # 증강된 이미지 저장 경로
output_label_file = r"C:\Users\edu31\Desktop\YoloV8\augmented_images\augmented_images_bottom\region_labels.csv"  # 증강된 라벨 저장 경로

# 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)

# 증강 횟수 설정
augmentation_count = 10

# 증강 파이프라인 설정
transform = Compose([
    HorizontalFlip(p=0.5),            # 좌우 반전
    RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 조정
    Rotate(limit=30, p=0.5),          # 회전
    RandomScale(scale_limit=0.2, p=0.5)  # 크기 조정
], bbox_params=BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

# CSV 파일 불러오기
df = pd.read_csv(csv_file)

# 증강된 데이터 저장 목록
augmented_data = []

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
    labels = [row['라벨']]

    # 증강 수행
    for i in range(augmentation_count):
        try:
            augmented = transform(image=image, bboxes=bboxes, labels=labels)

            # 유효한 바운딩 박스만 처리
            valid_bboxes = []
            valid_labels = []
            for bbox, label in zip(augmented['bboxes'], augmented['labels']):
                if 0 <= bbox[0] < width and 0 <= bbox[1] < height and 0 < bbox[2] <= width and 0 < bbox[3] <= height:
                    valid_bboxes.append(bbox)
                    valid_labels.append(label)
                else:
                    print(f"좌표 벗어난 바운딩 박스 제거됨: {bbox}")

            if not valid_bboxes:  # 유효한 바운딩 박스가 없으면 패스
                continue

            # 증강된 이미지 저장
            output_image_name = f"{os.path.splitext(row['이미지'])[0]}_aug_{i+1}.jpg"
            output_image_path = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_path, augmented['image'])

            # 증강된 바운딩 박스 저장
            for bbox, label in zip(valid_bboxes, valid_labels):
                augmented_data.append({
                    "이미지": output_image_name,
                    "X1": int(bbox[0]),
                    "Y1": int(bbox[1]),
                    "X2": int(bbox[2]),
                    "Y2": int(bbox[3]),
                    "라벨": label
                })
        except Exception as e:
            print(f"증강 중 오류 발생: {e}")

# 증강된 라벨 CSV 파일 저장
if augmented_data:
    pd.DataFrame(augmented_data).to_csv(output_label_file, index=False)
    print("이미지 증강 및 라벨 저장 완료!")
else:
    print("유효한 증강 데이터가 없습니다.")
