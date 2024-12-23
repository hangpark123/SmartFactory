import serial
import time

# 시리얼 포트 설정 (포트 이름과 보드레이트는 환경에 맞게 조정)
ser = serial.Serial('COM4', 9600)  # Windows의 경우 'COM4', Linux/Mac의 경우 '/dev/ttyUSB0' 등

time.sleep(2)  # 아두이노 리셋 시간 대기

def send_command(command):
    ser.write((command + '\n').encode())  # 명령어 전송
    time.sleep(1)  # 명령어 처리 시간 대기

try:
    while True:
        user_input = input("명령을 입력하세요 (1: 시계 방향 90도, 2: 반시계 방향 90도, q: 종료): ")
        if user_input == 'q':
            break
        elif user_input in ['1', '2']:
            send_command(user_input)
        else:
            print("잘못된 입력입니다. 다시 시도하세요.")
finally:
    ser.close()  # 시리얼 포트 닫기
