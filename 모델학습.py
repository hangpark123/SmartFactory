from ultralytics import YOLO

if __name__ == "__main__":
    # YOLOv8 모델 초기화
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano 모델 사용

    # 모델 학습
    model.train(
        data=r"C:\Users\edu31\Desktop\YoloV8\data.yaml",  # 데이터셋 YAML 파일 경로
        epochs=100,                                      # 학습 에폭 수
        imgsz=640,                                       # 입력 이미지 크기
        batch=16,                                        # 배치 사이즈
        name="12_17_AM10",                            # 결과 폴더 이름
        project=r"C:\Users\edu31\Desktop\YoloV8\trained_models"  # 학습 모델 저장 경로
    )

    print("모델 학습이 완료되었습니다!")
