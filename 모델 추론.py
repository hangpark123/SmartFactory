import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO(r"runs/detect/train2/weights/best.pt")  # 학습된 모델 경로

# 비디오 파일 열기
video_path = r"C:\Users\edu31\Desktop\YoloV8\video\123.mp4"  # 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

# 비디오 출력 설정
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 출력 포맷
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# 비디오 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model(frame)  # 모델을 사용하여 프레임에서 객체 탐지

    # 탐지된 객체 표시
    for result in results:
        boxes = result.boxes  # 바운딩 박스 정보
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            label = box.cls[0]

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(label)]} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과를 비디오로 저장
    out.write(frame)

    # 결과 영상 출력
    cv2.imshow("Detection", frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
