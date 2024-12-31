# YOLOv8 프로젝트

YOLOv8은 객체 탐지를 위한 최신 모델로, 실시간 애플리케이션에서 우수한 성능을 자랑합니다. 이 프로젝트는 YOLOv8 모델을 활용하여 객체 탐지, 데이터 증강, 라벨링, 모델 학습 및 예측을 수행할 수 있는 다양한 도구를 제공합니다.

---

## 📂 프로젝트 디렉토리 구조

- **`augmented_images/`**: 데이터 증강 후 생성된 이미지들이 저장됩니다.
- **`camera_snapshots/`**: 카메라로 촬영된 스냅샷 이미지.
- **`captured_images/`**: 캡처된 원본 이미지.
- **`captured_objects/`**: 탐지된 객체 이미지.
- **`csv/`**: 데이터와 관련된 CSV 파일.
- **`dataset_camera1/`**, **`dataset_camera2/`**: 각 카메라로부터 수집된 데이터셋.
- **`labels/`**: 이미지 라벨 파일.
- **`templates/`**: 템플릿 파일.
- **`trained_models/`**: 학습된 YOLOv8 모델 파일.
- **`video/`**: 비디오 파일.

---

## 🛠 주요 스크립트

- **`app3.py`**: 메인 애플리케이션 스크립트.
- **`predict.py`**: 학습된 모델로 객체 탐지 실행.
- **`resnet.py`**: ResNet 모델 관련 기능.
- **`데이터증강.py`**: 이미지 증강.
- **`라벨링.py`**: 이미지 라벨링 도구.
- **`사진찍자.py`**: 카메라로 이미지 캡처.
- **`모델학습_카메라1.py`**, **`모델학습_카메라2.py`**: 데이터 기반 모델 학습.
- **`욜로_데이터증강_csv변환.py`**: YOLO 형식 변환 및 데이터 증강.

---

## 🚀 설치 및 실행 방법

### 1️⃣ 필수 패키지 설치

프로젝트 실행에 필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비

`dataset_camera1/` 또는 `dataset_camera2/` 디렉토리에 데이터를 추가합니다.

### 3️⃣ 모델 학습

아래 명령어를 사용하여 모델을 학습시킵니다:

```bash
python 모델학습_카메라1.py
# 또는
python 모델학습_카메라2.py
```

### 4️⃣ 예측 수행

학습된 모델로 객체 탐지를 수행합니다:

```bash
python predict.py --source path_to_image_or_video
```

---

## 📋 참고 자료

- [YOLOv8 공식 문서](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md)

---

## 🤝 기여 방법

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4. 푸시합니다 (`git push origin feature/AmazingFeature`).
5. 풀 리퀘스트를 생성합니다.

---

## 📜 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.

