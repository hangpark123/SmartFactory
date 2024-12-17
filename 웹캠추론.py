if __name__ == '__main__':
    from ultralytics import YOLO

    # 학습된 모델 로드
    model = YOLO(r"C:\Users\edu31\Desktop\YoloV8\trained_models\custom_model42\weights\best.pt")

    # 웹캠 추론 수행
    model.predict(source=0, show=True)
