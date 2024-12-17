import pandas as pd

# CSV 파일 로드
csv_file = r"C:\Users\edu31\Desktop\YoloV8\region_labels.csv"
df = pd.read_csv(csv_file)

# 부정적인 좌표 확인
invalid_labels = df[(df['X1'] < 0) | (df['Y1'] < 0) | (df['X2'] < 0) | (df['Y2'] < 0)]

if not invalid_labels.empty:
    print("부정적 좌표 발견:", invalid_labels)
else:
    print("모든 라벨이 정상입니다.")
