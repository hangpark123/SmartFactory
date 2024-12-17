import cv2
import os
import pandas as pd

# 설정
image_folder = r"C:\Users\edu31\Desktop\YoloV8\captured_images\captured_images_bottom"  # 라벨링할 이미지 불러오기 경로
output_file = r"C:\Users\edu31\Desktop\YoloV8\labeled_images\labeled_images_bottom\region_labels.csv"  # CSV 라벨 파일 저장 경로
output_image_dir = r"C:\Users\edu31\Desktop\YoloV8\labeled_images\labeled_images_bottom"  # 라벨링된 이미지 저장 경로

# 저장 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)

# 이미지 목록 불러오기
images = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
current_index = 0

# 상태 변수
start_x, start_y = -1, -1
end_x, end_y = -1, -1
drawing = False
current_image = None
labels = []

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
            label = "part"
            labels.append((images[current_index], start_x, start_y, end_x, end_y, label))
            print(f"이미지: {images[current_index]}, 좌표: ({start_x}, {start_y}) ~ ({end_x}, {end_y}), 라벨: {label}")

            # 바운딩 박스가 그려진 이미지 저장
            labeled_image = current_image.copy()
            cv2.rectangle(labeled_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            output_path = os.path.join(output_image_dir, images[current_index])
            cv2.imwrite(output_path, labeled_image)
            print(f"라벨링된 이미지 저장: {output_path}")

        # 다음 이미지 로드
        current_index += 1
        if current_index < len(images):
            load_next_image()
        else:
            save_labels()
            print("라벨링이 완료되었습니다.")
            cv2.destroyAllWindows()

# 다음 이미지 로드
def load_next_image():
    global current_image
    if current_index < len(images):
        current_image = cv2.imread(os.path.join(image_folder, images[current_index]))
        cv2.imshow("이미지 라벨링", current_image)

# 라벨 저장
def save_labels():
    pd.DataFrame(labels, columns=["이미지", "X1", "Y1", "X2", "Y2", "라벨"]).to_csv(output_file, index=False)
    print(f"라벨 CSV 파일 저장 완료: {output_file}")

# OpenCV 윈도우 생성 및 마우스 이벤트 설정
cv2.namedWindow("이미지 라벨링")
cv2.setMouseCallback("이미지 라벨링", mouse_callback)

# 첫 이미지 로드
if len(images) > 0:
    load_next_image()
    cv2.waitKey(0)
else:
    print("이미지가 없습니다.")
