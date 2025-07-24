# Qc_MartialArts_NN
Attempt at a neural network
1.) The test2 file contains working code for a different type of neural network; its purpose is to be studied and learned from, it is directly from pytorch website (we will be adapting this code later on)
2.) Here are how the edges correspond to each number keypoint 

(0, 1): Nose → Left Eye

(0, 2): Nose → Right Eye

(1, 3): Left Eye → Left Ear

(2, 4): Right Eye → Right Ear

(0, 5): Nose → Left Shoulder

(0, 6): Nose → Right Shoulder

(5, 7): Left Shoulder → Left Elbow

(7, 9): Left Elbow → Left Wrist

(6, 8): Right Shoulder → Right Elbow

(8, 10): Right Elbow → Right Wrist

(5, 11): Left Shoulder → Left Hip

(6, 12): Right Shoulder → Right Hip

(11, 13): Left Hip → Left Knee

(13, 15): Left Knee → Left Ankle

(12, 14): Right Hip → Right Knee

(14, 16): Right Knee → Right Ankle