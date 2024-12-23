import cv2
import time
from ultralytics import YOLO
import os

# YOLO 모델 초기화
model = YOLO(r'C:\Users\edu31\YoloV8\trained_models\custom_model10\weights\best.pt') 

# 경로 설정
video_source = 1  # 웹캠 사용 (또는 동영상 파일 경로)
output_dir = r"C:\Users\edu31\YoloV8\captured_objects"
os.makedirs(output_dir, exist_ok=True)

# 트래커 초기화
active_objects = set()  # 현재 화면에 있는 객체 추적
last_seen = {}  # 마지막으로 본 객체 시간 기록
DISAPPEAR_THRESHOLD = 3  # 초 (사라짐으로 간주하는 시간)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_source)

# 프레임 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model.predict(source=frame, save=False, conf=0.2)

    # 현재 프레임에서 탐지된 객체 관리
    current_objects = set()

    for idx, result in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, cls = result.tolist()
        cls = int(cls)

        # 라운딩 박스 그리기
        label = f"Class {cls} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 객체가 이전에 감지되지 않았으면 저장
        if cls not in active_objects:
            # 좌표 변환 및 객체 이미지 자르기
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = frame[y1:y2, x1:x2]

            # 이미지 저장
            output_file = os.path.join(output_dir, f"object_class_{cls}_{time.time():.0f}.jpg")
            cv2.imwrite(output_file, cropped_img)
            print(f"객체 저장됨: {output_file}")

            # 객체 등록
            active_objects.add(cls)

        # 현재 객체 목록에 추가
        current_objects.add(cls)
        last_seen[cls] = time.time()

    # 오래된 객체 제거 (화면에서 사라짐 판정)
    for obj in list(active_objects):
        if obj not in current_objects and time.time() - last_seen[obj] > DISAPPEAR_THRESHOLD:
            print(f"객체 {obj} 제거됨")
            active_objects.remove(obj)

    # 화면에 출력
    cv2.imshow("객체 탐지", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리 및 종료
cap.release()
cv2.destroyAllWindows()
