import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO(r'C:\Users\edu31\YoloV8\trained_models\camera_2_test5\weights\best.pt')  # 자신의 학습된 모델 경로

# 웹캠 열기 (2번 장치는 camera_1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # YOLOv8을 통한 객체 인식, 신뢰도 임계값 설정
    results = model.predict(frame, conf=0.85)  # 신뢰도 임계값을 0.75로 설정

    # 결과를 프레임 위에 그리기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 USB Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
