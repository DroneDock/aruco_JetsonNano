"""
CAMERA CALIBRATION PART (B) 

This script generates data in the form of images.
Prior to running this script, we will need to print the ArUco board as detailed in 'aruco_board_generation.py'.

    1) Provide desired path to store images.
    2) Press 'c' to capture image and display it.
        a) Capture the ArUco board from various angles for better calibration purposes.
    3) Press any button to continue.
    4) Press 'q' to quit.

Created by: Jalen
"""

import cv2
import time
from imutils.video import VideoStream

# Initialize the camera using Imutils VideoStream function (Note that the VideoCapture function does not work for Jetson Nano)
camera = VideoStream("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)500, height=(int)500,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink").start()
time.sleep(2)

# Path to store data in the form of images
path = "/home/gai/aruco_markers/camera_calibration/aruco_calibration_data"

# Capture image and save it in the path
count = 0
while True:
    frame = camera.read()
    name = path + str(count)+".jpg"
    frame = camera.read()
    cv2.imshow("img", frame)


    if cv2.waitKey(20) & 0xFF == ord('c'):
        cv2.imwrite(name, frame)
        cv2.imshow("img", frame)
        count += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):

            break;
