import os
import pandas as pd
import shutil

# 경로 설정
csv_file = r"C:\Users\edu31\Desktop\YoloV8\region_labels.csv"
image_dir = r"C:\Users\edu31\Desktop\YoloV8\captured_images"
output_dir = r"C:\Users\edu31\Desktop\YoloV8\dataset"

# 데이터셋 폴더 생성
train_image_dir = os.path.join(output_dir, "images", "train")
train_label_dir = os.path.join(output_dir, "labels", "train")
val_image_dir = os.path.join(output_dir, "images", "val")
val_label_dir = os.path.join(output_dir, "labels", "val")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 클래스 맵핑 설정
class_map = {"part": 0}  # 필요한 클래스와 ID 매핑

# 데이터셋 변환
image_files = df['이미지'].unique()
train_ratio = 0.8
num_train = int(len(image_files) * train_ratio)

for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    if not os.path.exists(image_path):
        print(f"{image_file}을 찾을 수 없습니다.")
        continue

    # YOLO 형식 라벨 생성
    image_width, image_height = 640, 480  # 이미지 크기 (수동 설정 필요)

    # 라벨 파일 생성
    label_file = f"{os.path.splitext(image_file)[0]}.txt"
    label_data = df[df['이미지'] == image_file]

    # YOLO 라벨 변환
    with open(os.path.join(train_label_dir if i < num_train else val_label_dir, label_file), "w") as f:
        for _, row in label_data.iterrows():
            x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
            class_name = row['라벨']

            if class_name not in class_map:
                continue

            class_id = class_map[class_name]

            # YOLO 형식 변환
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 이미지 파일 복사
    shutil.copy(image_path, train_image_dir if i < num_train else val_image_dir)

print("데이터셋 변환 완료.")
