import os
import pandas as pd

# 경로 설정
csv_file = r"C:\Users\edu31\Desktop\YoloV8\augmented_images\augmented_images_bottom\region_labels.csv"
image_dir = r"C:\Users\edu31\Desktop\YoloV8\augmented_images\augmented_images_bottom"
output_label_dir = r"C:\Users\edu31\Desktop\YoloV8\labels\bottom"

# 저장 폴더 생성
os.makedirs(output_label_dir, exist_ok=True)

# CSV 파일 불러오기
df = pd.read_csv(csv_file)

# 이미지 크기 (예시, 실제 이미지 크기를 불러와야 정확함)
image_width = 640
image_height = 480

# 라벨 변환
for _, row in df.iterrows():
    image_name = row['이미지']
    x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
    label = row['라벨']

    # 클래스 ID (예: "part" → 0으로 설정)
    class_id = 0

    # 바운딩 박스 좌표를 YOLO 형식으로 변환
    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    # 라벨 파일 저장
    label_file = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("CSV 파일이 YOLO 형식으로 변환되었습니다.")
