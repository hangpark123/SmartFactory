import cv2
import os
import pandas as pd

# 설정
image_folder = r"C:\Users\edu31\Pictures\Camera Roll"
output_image_dir = r"C:\Users\edu31\YoloV8\labels\camera_2"
output_label_dir = os.path.join(output_image_dir, "labels")

# 저장 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 이미지 목록 불러오기
images = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
current_index = 0

# 상태 변수
start_x, start_y = -1, -1
end_x, end_y = -1, -1
drawing = False
current_image = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, current_image, current_index

    if event == cv2.EVENT_LBUTTONDOWN:  # 드래그 시작
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # 드래그 중
        if drawing:
            temp_image = current_image.copy()
            cv2.rectangle(temp_image, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("이미지 라벨링", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # 드래그 끝
        drawing = False
        end_x, end_y = x, y

        # 선택 영역이 유효한 경우 저장
        if start_x != end_x and start_y != end_y:
            save_label_and_image(start_x, start_y, end_x, end_y)
            print(f"라벨 저장 및 이미지 저장 완료: {images[current_index]}")

        # 다음 이미지 로드
        current_index += 1
        if current_index < len(images):
            load_next_image()
        else:
            print("라벨링이 완료되었습니다.")
            cv2.destroyAllWindows()

# 다음 이미지 로드
def load_next_image():
    global current_image
    if current_index < len(images):
        current_image = cv2.imread(os.path.join(image_folder, images[current_index]))
        cv2.imshow("이미지 라벨링", current_image)

# 라벨 및 이미지 저장
def save_label_and_image(x1, y1, x2, y2):
    global current_image, current_index

    # YOLOv8 형식 좌표 계산
    img_h, img_w = current_image.shape[:2]
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h

    # 파일 이름 생성
    image_name = f"camera_2_{current_index+1:03}.jpg"
    txt_file = os.path.join(output_label_dir, f"{os.path.splitext(image_name)[0]}.txt")

    # 라벨 파일 저장
    with open(txt_file, "w") as f:
        f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 라벨링된 이미지 저장
    output_image_path = os.path.join(output_image_dir, image_name)
    labeled_image = current_image.copy()
    cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_image_path, labeled_image)

    print(f"라벨 TXT 저장: {txt_file}")
    print(f"라벨링된 이미지 저장: {output_image_path}")

# OpenCV 윈도우 생성 및 마우스 이벤트 설정
cv2.namedWindow("이미지 라벨링")
cv2.setMouseCallback("이미지 라벨링", mouse_callback)

# 첫 이미지 로드
if len(images) > 0:
    load_next_image()
    cv2.waitKey(0)
else:
    print("이미지가 없습니다.")
