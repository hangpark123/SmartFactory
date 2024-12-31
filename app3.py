import os
import time
from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from predict import predict_image  # CNN 예측 함수 가져오기
import threading

# YOLO 모델 로드 (카메라별로 다른 모델)
model_camera_1 = YOLO(r"C:\\Users\\edu31\\YoloV8\\trained_models\\camera_2_test5\\weights\\best.pt", verbose=False)
model_camera_2 = YOLO(r"C:\\Users\\edu31\\YoloV8\\trained_models\\camera_1_test15\\weights\\best.pt", verbose=False)


# Flask 앱 생성
app = Flask(__name__)

# 특정 영역 정의 (예: 물체 검출 영역)
counts = {'pass': 0, 'fail': 0}
snapshot_dir = r"C:\\Users\\edu31\\YoloV8\\camera_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

def process_snapshot_async(snapshot_path):
    """스냅샷 저장 후 CNN 예측을 비동기적으로 실행"""
    global counts
    try:
        predicted_label, class_name = predict_image(snapshot_path)
        print(f"클래스 {predicted_label}, 클래스 이름: {class_name}")

        if class_name in counts:
            counts[class_name] += 1
    except Exception as e:
        print(f"비동기 CNN 예측 중 오류: {e}")

def gen_camera_1():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("카메라 1번을 열 수 없습니다.")
        return

    prev_time = 0
    fps = 15  # 초당 15프레임 처리

    while True:
        success, frame = camera.read()
        if not success:
            print("카메라 1번에서 프레임을 읽을 수 없습니다. 재시도 중...")
            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(0)
            continue

        # 현재 시간 계산
        curr_time = time.time()
        if (curr_time - prev_time) < (1.0 / fps):
            continue  # FPS 제한 조건을 충족하지 못하면 루프 진행

        prev_time = curr_time

        # YOLOv8 객체 검출을 위한 크기 조정 (640x480)
        yolo_frame = cv2.resize(frame, (640, 480))

        # YOLOv8 객체 검출
        results = model_camera_1.predict(yolo_frame, conf=0.6)

        # 감지된 객체 처리
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf.item()

            # 바운딩 박스 추가
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(yolo_frame, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임을 JPEG로 인코딩하여 클라이언트에 전송
        _, buffer = cv2.imencode('.jpg', yolo_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_camera_2():
    camera = cv2.VideoCapture(2)
    if not camera.isOpened():
        print("카메라 2번을 열 수 없습니다.")
        return

    global counts
    prev_time = 0
    fps = 10  # 초당 10프레임 처리
    snapshot_saved = False  # 스냅샷 저장 여부를 추적하는 변수

    while True:
        success, frame = camera.read()
        if not success:
            print("카메라 2번에서 프레임을 읽을 수 없습니다. 재시도 중...")
            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(2)
            continue

        # 현재 시간 계산
        curr_time = time.time()
        if (curr_time - prev_time) < (1.0 / fps):
            continue  # FPS 제한 조건을 충족하지 못하면 루프 진행

        prev_time = curr_time

        # 프레임 크기 조정 (640x480)
        yolo_frame = cv2.resize(frame, (640, 480))
        frame_height, frame_width, _ = frame.shape

        # YOLOv8 객체 검출
        results = model_camera_2.predict(yolo_frame, conf=0.85)

        # 감지된 객체 처리
        if results[0].boxes:
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf.item()

                # 객체 중심 계산
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 객체 중심이 화면 중앙에 가깝거나 부드러운 범위에 있을 때 스냅샷 저장
                if (frame_width // 4 < center_x < 3 * frame_width // 4 and
                        frame_height // 4 < center_y < 3 * frame_height // 4):
                    # 바운딩 박스 추가
                    cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(yolo_frame, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if not snapshot_saved:
                        snapshot_path = os.path.join(snapshot_dir, f"camera2_snapshot_{int(time.time())}.jpg")
                        cv2.imwrite(snapshot_path, frame)  # 원본 크기의 프레임 저장
                        print(f"카메라 2 스냅샷 저장됨: {snapshot_path}")
                        snapshot_saved = True

                        # 비동기 CNN 호출
                        threading.Thread(target=process_snapshot_async, args=(snapshot_path,)).start()
                else:
                    snapshot_saved = False  # 중심 조건을 만족하지 못하면 초기화
        else:   
            snapshot_saved = False  # 객체가 없으면 초기화

        # 프레임을 JPEG로 인코딩하여 클라이언트에 전송
        _, buffer = cv2.imencode('.jpg', yolo_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera_1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    return Response(gen_camera_2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    """정상품 및 불량품 카운트 반환"""
    return jsonify(counts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
