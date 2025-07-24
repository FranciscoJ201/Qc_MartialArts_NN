import os
import shutil

def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and all contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Example usage:
def clear_all():
    directory = 'output_plots'  # replace with your directory path
    # directory2 = 'vid_output'
    clear_directory(directory)
    # clear_directory(directory2)
clear_all()