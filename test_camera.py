"""
This script allows the user to test the camera's functionalities when implemented with cv2.

Created by: Jalen
"""

# METHOD 1 - VIDEO CAPTURE -------------------------------------------------------------------------
import sys
import cv2

def read_cam():
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)500, height=(int)500,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink")
    if cap.isOpened():
        cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("demo", 100, 100)
        while True:
            ret_val, img = cap.read();
            cv2.imshow('demo',img)
            # cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
     print ("camera open failed")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_cam()



# METHOD 2 - NANOCAMERA -------------------------------------------------------------------------
# import cv2
# import nanocamera as nano

# if __name__ == '__main__':
#     # Create the Camera instance
#     camera = nano.Camera(flip=0, width=640, height=480, fps=30)
#     print('CSI Camera ready? - ', camera.isReady())
#     while camera.isReady():
#         try:
#             # read the camera image
#             frame = camera.read()
#             # display the frame
#             cv2.imshow("Video Frame", frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         except KeyboardInterrupt:
#             break

#     # close the camera instance
#     camera.release()

#     # remove camera object
#     del camera


# # METHOD 3 - GSTREAMER-PIPELINE ------------------------------------------------------------------
# # MIT License
# # Copyright (c) 2019-2022 JetsonHacks

# # Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# # NVIDIA Jetson Nano Developer Kit using OpenCV
# # Drivers for the camera and OpenCV are included in the base image

# import cv2

# """ 
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of each camera pane in the window on the screen
# Default 1920x1080 displayd in a 1/4 size window
# """

# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1920,
#     capture_height=1080,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc sensor-id=%d ! "
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             sensor_id,
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )


# def show_camera():
#     window_title = "CSI Camera"

#     # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
#     print(gstreamer_pipeline(flip_method=0))
#     video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#     if video_capture.isOpened():
#         try:
#             window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
#             while True:
#                 ret_val, frame = video_capture.read()
#                 # Check to see if the user closed the window
#                 # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
#                 # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
#                 if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
#                     cv2.imshow(window_title, frame)
#                 else:
#                     break 
#                 keyCode = cv2.waitKey(10) & 0xFF
#                 # Stop the program on the ESC key or 'q'
#                 if keyCode == 27 or keyCode == ord('q'):
#                     break
#         finally:
#             video_capture.release()
#             cv2.destroyAllWindows()
#     else:
#         print("Error: Unable to open camera")


# if __name__ == "__main__":
#     show_camera()
