import cv2
import os
from os import listdir
from concurrent.futures import ThreadPoolExecutor

def make_video(name, original_video_path):
    image_folder = 'AlphaPose_Code/output_plots'
    output_video_path = f'Video_Outputs/{name}.mp4'

    def extract_frame_number(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except:
            return -1

    # Sort image files
    image_files = sorted(
        [img for img in listdir(image_folder) if img.endswith(".png")],
        key=extract_frame_number
    )

    if not image_files:
        print("No images found.")
        return

    # Get video resolution from first image
    first_image_path = os.path.join(image_folder, image_files[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape

    # Get FPS from original video
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Use faster codec (optional: 'XVID' or 'MJPG' is faster than 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Precompute image paths
    image_paths = [os.path.join(image_folder, f) for f in image_files]

    # Load all frames in parallel
    def load_image(path):
        return cv2.imread(path)

    with ThreadPoolExecutor() as executor:
        for frame in executor.map(load_image, image_paths):
            if frame is not None:
                video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved to {output_video_path} at {fps:.2f} FPS")
