from ultralytics import YOLO
import os

def train_model(model, yaml_file, project_dir, max_epochs=1000):
    try:
        print(f"✅ 모델 학습 시작: 최대 {max_epochs} 에폭")
        model.train(
            data=yaml_file,         
            epochs=max_epochs,      
            imgsz=640,              
            batch=16,               
            name="camera_2_test",    
            project=project_dir,    
            device=0,  # CUDA 디바이스 설정 (0은 첫 번째 GPU)
            verbose=True            
        )
        print("✅ 모델 학습이 완료되었습니다!")
    except Exception as e:
        print(f"❌ 모델 학습 중 오류 발생: {e}")
        model.save(os.path.join(project_dir, "custom_model_last.pt"))
        print("✅ 모델이 오류 발생 시 저장되었습니다.")

# 메인 실행부
if __name__ == "__main__":
    # 모델 초기화
    model = YOLO("yolov8n.pt")  

    # 경로 설정
    yaml_file = r"C:\Users\edu31\YoloV8\data_camera2.yaml"
    project_dir = r"C:\Users\edu31\YoloV8\trained_models"

    # 모델 학습 시작
    train_model(model, yaml_file, project_dir)
