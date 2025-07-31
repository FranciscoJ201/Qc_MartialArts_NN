import cv2
import os
from os import listdir
def make_video(name):
    # Directory where your plots are saved
    image_folder = 'OpenPose_Code/newplots'
    output_video_path = f'OpenPose_Code/Video_Outputs/{name}.mp4'


    # Get all image filenames and sort them by frame number
    def extract_frame_number(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except:
            return -1

    images = sorted(
        [img for img in listdir(image_folder) if img.endswith(".png")],
        key=extract_frame_number
    )

    # Read the first image to get frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define video writer (you can change fps and codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # or adjust to match original video speed
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")