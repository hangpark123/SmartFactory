camera = cv2.VideoCapture(1)
if not camera.isOpened():
    print("카메라를 열 수 없습니다. 장치 번호를 확인하세요.")
else:
    print("카메라가 정상적으로 열렸습니다.")
