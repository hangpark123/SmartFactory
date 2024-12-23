import os
import pandas as pd

# 경로 설정
label_dir = r"C:\Users\edu31\YoloV8\labels\camera_1\labels"
output_csv_file = r"C:\Users\edu31\YoloV8\labels\camera_1.csv"
image_width = 1920  # 이미지 너비
image_height = 1080  # 이미지 높이

# 변환 결과 저장 목록
converted_data = []

# 모든 라벨 파일 변환
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        image_name = f"{os.path.splitext(label_file)[0]}.jpg"
        label_file_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_file_path):
            print(f"라벨 파일을 찾을 수 없습니다: {label_file_path}")
            continue

        with open(label_file_path, "r") as f:
            lines = f.readlines()
            if not lines:
                print(f"{label_file_path} 파일이 비어 있습니다.")
                continue

            for line in lines:
                try:
                    label, x_center, y_center, box_width, box_height = map(float, line.split())

                    # YOLO 형식을 Pascal VOC 형식으로 변환
                    x1 = int((x_center - box_width / 2) * image_width)
                    y1 = int((y_center - box_height / 2) * image_height)
                    x2 = int((x_center + box_width / 2) * image_width)
                    y2 = int((y_center + box_height / 2) * image_height)

                    # 변환 결과 추가
                    converted_data.append({
                        "이미지": image_name,
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                        "라벨": "camera_1"
                    })
                    print(f"추가된 데이터: {converted_data[-1]}")

                except Exception as e:
                    print(f"변환 중 오류 발생: {e} | 파일: {label_file}")

# CSV 파일 저장
df = pd.DataFrame(converted_data)
if not df.empty:
    df.to_csv(output_csv_file, index=False, encoding="utf-8-sig")
    print(f"변환 완료! CSV 파일 저장됨: {output_csv_file}")
else:
    print("CSV 파일에 추가할 데이터가 없습니다.")
