import cv2
import os

# 저장 폴더 생성
save_folder = "captured_images\captured_images_bottom"
os.makedirs(save_folder, exist_ok=True)

# 카메라 초기화
camera = cv2.VideoCapture(1)  # USB 카메라가 기본 장치로 설정됨

# 해상도 설정 (필요 시 조정)

num_photos = 100
delay_between_photos = 0.01  # 초 단위 대기 시간

for i in range(num_photos):
    ret, frame = camera.read()
    if not ret:
        print("카메라에서 이미지를 가져오지 못했습니다.")
        break

    # 이미지 저장
    file_name = os.path.join(save_folder, f"photo_{i+1:03}.jpg")
    cv2.imwrite(file_name, frame)
    print(f"{file_name} 저장 완료.")

# 카메라 및 리소스 해제
camera.release()
cv2.destroyAllWindows()
