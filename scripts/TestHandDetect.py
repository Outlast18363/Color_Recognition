# import the necessary packages
import datetime
from threading import Thread
import cv2

import mediapipe as mp
import math
import numpy as np
from profile_utils import *
from hand_utils import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")

ap.add_argument("-c", "--camera", type=str, default="csi",
    help="Type of camera to use")

args = vars(ap.parse_args())

# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")
hand_sol = mp.solutions.hands
hand_proc = hand_sol.Hands(max_num_hands=1)
hand_drawer = mp.solutions.drawing_utils
hand_detect_cnt = 0

stream = cv2.VideoCapture(args["camera"])
fps = FPS().start()
# loop over some frames
while fps._numFrames < args["num_frames"]:
    # grab the frame from the stream and resize it to have a maximum
    # width of 400 pixels
    (grabbed, frame) = stream.read()
    if frame is not None:
        frame, results = procFrame(frame, hand_sol, hand_proc, hand_drawer)
        if results is not None: hand_detect_cnt += 1
    
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] detecte hand: {:3d}".format(hand_detect_cnt))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()