import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate, RandomScale, BboxParams

# 경로 설정
image_path = r"C:\Users\edu31\Desktop\YoloV8\image\123.jpg"
xml_path = r"C:\Users\edu31\Desktop\YoloV8\annotations\123.xml"
output_dir = r"C:\Users\edu31\Desktop\YoloV8\augmented"
yolo_data_dir = r"C:\Users\edu31\Desktop\YoloV8\annotations"
num_augmentations = 5

# 이미지 증강 함수
def augment_image(image, bboxes):
    bbox_params = BboxParams(format='pascal_voc', label_fields=['class_labels'])
    
    transform = Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Rotate(limit=40, p=0.5),
        RandomScale(scale_limit=0.2, p=0.5)
    ], bbox_params=bbox_params)

    augmented = transform(image=image, bboxes=bboxes, class_labels=[0] * len(bboxes))  # 클래스 라벨 예시
    return augmented['image'], augmented['bboxes']

# YOLO 형식으로 변환하는 함수
def create_yolo_labels(image_path, xml_path, output_dir):
    # XML 파싱
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 이미지 크기 정보
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # 클래스 라벨 목록 (예시로 단일 클래스 사용)
    class_map = {"class_name": 0}  # 클래스 이름과 ID 매핑

    # 라벨 정보 추출
    bboxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in class_map:
            continue
        label_id = class_map[label]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # YOLO 형식으로 변환 (상대적 좌표)
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        bboxes.append([x_center, y_center, w, h])
        labels.append(label_id)

    # YOLO 라벨 파일 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = os.path.join(output_dir, f"{base_name}.txt")
    with open(label_file, "w") as f:
        for bbox, label in zip(bboxes, labels):
            f.write(f"{label} {' '.join(map(str, bbox))}\n")

    # 이미지는 다른 경로로 복사 (증강된 이미지 경로로 복사)
    augmented_image_path = os.path.join(output_dir, f"{base_name}.jpg")
    shutil.copy(image_path, augmented_image_path)

# 증강된 이미지와 XML을 YOLO 형식으로 변환 후 저장
def augment_and_save_yolo_format(image_path, xml_path, output_dir, num_augmentations):
    os.makedirs(output_dir, exist_ok=True)

    # 원본 이미지를 불러옴
    image = cv2.imread(image_path)

    # XML 파싱
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 원본 바운딩 박스 좌표 추출
    bboxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])

    for i in range(num_augmentations):
        # 증강된 이미지 생성
        augmented_image, augmented_bboxes = augment_image(image, bboxes)

        # 증강된 이미지와 XML 경로 설정
        augmented_image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i+1}.jpg")
        augmented_xml_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(xml_path))[0]}_aug_{i+1}.xml")

        # 증강된 이미지 저장
        cv2.imwrite(augmented_image_path, augmented_image)

        # 증강된 XML 파일 생성
        tree = ET.ElementTree(root)
        for obj, bbox in zip(root.findall("object"), augmented_bboxes):
            bndbox = obj.find("bndbox")
            bndbox.find("xmin").text = str(bbox[0])
            bndbox.find("ymin").text = str(bbox[1])
            bndbox.find("xmax").text = str(bbox[2])
            bndbox.find("ymax").text = str(bbox[3])
        tree.write(augmented_xml_path)

        # YOLO 라벨 파일 저장
        create_yolo_labels(augmented_image_path, augmented_xml_path, output_dir)

# 증강된 데이터셋 생성 및 저장
augment_and_save_yolo_format(image_path, xml_path, yolo_data_dir, num_augmentations)
