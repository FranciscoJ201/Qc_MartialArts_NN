import cv2
import os
from os import listdir

def make_video(name, original_video_path):
    image_folder = 'AlphaPose_Code/output_plots'
    output_video_path = f'Video_Outputs/{name}.mp4'

    def extract_frame_number(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except:
            return -1

    images = sorted(
        [img for img in listdir(image_folder) if img.endswith(".png")],
        key=extract_frame_number
    )

    if not images:
        print("No images found.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 👇 Detect FPS from the original input video
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path} at {fps:.2f} FPS")
