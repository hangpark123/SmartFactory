from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time
from ultralytics import YOLO
import serial

# YOLO 모델 로드 (카메라별로 다른 모델)
model_camera_1 = YOLO(r"C:\Users\edu31\YoloV8\trained_models\camera_2_test5\weights\best.pt")  # 카메라 2용 모델 경로
model_camera_2 = YOLO(r"C:\Users\edu31\YoloV8\trained_models\camera_1_test15\weights\best.pt")  # 카메라 1용 모델 경로

# Flask 앱 생성
app = Flask(__name__)

# 시리얼 포트 설정 (아두이노 연결)
ser = serial.Serial('COM4', 9600)
time.sleep(2)  # 아두이노 리셋 시간 대기

def send_command(command):
    """아두이노로 명령 전송"""
    ser.write((command + '\n').encode())
    time.sleep(1)  # 명령 처리 대기

# 특정 영역 정의 (예: 물체 검출 영역)
counts = {'normal': 0, 'defect': 0}
snapshot_dir = r"C:\Users\edu31\YoloV8\camera_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

def gen_camera_1():
    camera = cv2.VideoCapture(0)
    prev_time = 0
    fps = 10  # 초당 10프레임 처리

    while True:
        success, frame = camera.read()
        if not success:
            break

        # 현재 시간
        curr_time = time.time()
        if (curr_time - prev_time) > 1.0 / fps:
            prev_time = curr_time

            # 프레임 크기 조정 (예: 가로 640, 세로 480)
            frame = cv2.resize(frame, (640, 480))

            # YOLOv8 객체 검출 수행 (카메라 1 모델 사용)
            results = model_camera_1.predict(frame, conf=0.8)
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label_index = int(result.cls.item())  # 텐서를 정수로 변환
                confidence = result.conf.item()  # 텐서를 숫자로 변환

                # 바운딩 박스와 라벨 추가
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_index}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 프레임을 JPEG로 인코딩하여 클라이언트에 전송
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_camera_2():
    camera = cv2.VideoCapture(2)
    global counts
    snapshot_saved = False  # 스냅샷 저장 여부를 추적하는 변수

    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLOv8 객체 검출 (카메라 2 모델 사용)
        results = model_camera_2.predict(frame, conf=0.75)  # 신뢰도 임계값 설정

        # 감지된 객체가 있는지 확인
        if results[0].boxes:  # 감지된 바운딩 박스가 있으면
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label_index = int(result.cls.item())  # 텐서를 정수로 변환
                confidence = result.conf.item()  # 텐서를 숫자로 변환

                # 바운딩 박스와 라벨 추가
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_index}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 감지된 객체가 있으면 아두이노로 명령 전송
                print("Object detected by Camera 2")
                send_command('1')  # 시계 방향 90도 회전 명령

            # 스냅샷 저장 (한 번만 저장하도록 조건 추가)
            if not snapshot_saved:
                snapshot_path = os.path.join(snapshot_dir, f"camera2_snapshot_{int(time.time())}.jpg")
                cv2.imwrite(snapshot_path, frame)
                print(f"카메라 2 스냅샷 저장됨: {snapshot_path}")
                snapshot_saved = True  # 스냅샷이 저장되었음을 표시

        else:
            snapshot_saved = False  # 감지된 객체가 없으면 스냅샷 저장 상태 초기화

        # 프레임을 JPEG로 인코딩하여 클라이언트에 전송
        _, buffer = cv2.imencode('.jpg', frame)
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
    return jsonify(counts)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False로 변경
    finally:
        ser.close()