from reader import run_pose_plotter
from frameGUIandSelect import frame_selector
json,video,name = run_pose_plotter()
frame_selector(json,video)