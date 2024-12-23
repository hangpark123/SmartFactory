import os

# 설정
image_dir = r"C:\Users\edu31\Pictures\Camera Roll"
new_name_prefix = "camera_1_"

def rename_images(directory, prefix):
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for idx, filename in enumerate(image_files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}{idx:03}{os.path.splitext(filename)[1]}"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_filename}")

if __name__ == "__main__":
    rename_images(image_dir, new_name_prefix)
    print("이름 변경이 완료되었습니다.")
