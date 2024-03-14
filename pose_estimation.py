"""
This script detects the ArUco marker and pose estimate the translationa and rotational vectors of the marker respective to the camera.
This script can be divided into three sections:
    1) Definitions
        - Define the marker size and dictionary

    2) Load camera data
        - Load the camera matrix and distortion coefficients calculated and stored in the YAML file.

    3) Execution
        - Initialize the camera
        - Detect any ArUco marker present in the camera frame by drawing polylines.
        - Pose estimate and print out the translational and rotational values of the marker
        - Annotate the pose for better visualization purposes

Created by: Jalen
"""

# Standard Imports
import time
from pathlib import Path
import os

# Third-Party Imports
import cv2
import numpy as np
from imutils.video import VideoStream
import yaml

# Project-Specific Imports
from arucoDict import ARUCO_DICT



# DEFINITIONS ---------------------------------------------------------------------------------------------------------------------------------
# Marker
MARKER_SIZE = 60  # Square size [mm] - allow for pose and distance estimation
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_6X6_50"])
arucoParams = cv2.aruco.DetectorParameters_create()  # Use default parameters



# LOAD CAMERA DATA -----------------------------------------------------------------------------------------------------------------------------
# Load the camera matrix and distortion coefficients from YAML file
with open('camera_calibration_final/calibration.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
camMatrix = loadeddict.get('camera_matrix')
distCof = loadeddict.get('dist_coeff')
camMatrix = np.array(camMatrix)
distCof = np.array(distCof)

print("Loaded calibration data successfully")



# EXECUTION ------------------------------------------------------------------------------------------------------------
# Initialize the camera
vs = VideoStream("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)500, height=(int)500,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink").start()
time.sleep(2)

last_print_time = time.time()

# # Create a VideoWriter object to save the video
# output_folder = 'Videos'
# os.makedirs(output_folder, exist_ok=True)
# output_path = os.path.join(output_folder, 'test.avi')

# # 10s framerate, 1000x800 resolution
# result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (1000, 800))     

while True:
    
    frame = vs.read()
    # MUST be similar resolution to videowriter object
    frame = cv2.resize(frame, (1000, 800))      
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image=gray_frame,
                                                       dictionary=arucoDict,
                                                       parameters=arucoParams)

    # If ArUco marker is detected
    if corners:
        rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners, markerLength=MARKER_SIZE, cameraMatrix=camMatrix, distCoeffs=distCof
        )

        total_markers = range(0, ids.size)

        # Print pose estimation values every 2s for each marker
        current_time = time.time()
        if current_time - last_print_time >= 2.0:
            for markerID, i in zip(ids, total_markers):
                print(f"Marker ID: {markerID}")
                print(f"Translation Vector (tvec): {tVec[i].flatten()}")

                # print("Translation Vector (tvec):")
                # for value in tVec.flatten():
                #     print(f"    {value}")
                
                # print("Rotation Vector (rvec):")
                # for value in rVec.flatten():
                #     print(f"    {value}")
                
                print("-----------------------------")
                last_print_time = current_time

        
        for markerID, corner, i in zip(ids, corners, total_markers):

            topLeft, topRight, btmRight, btmLeft = corner.reshape((4, 2))

            # Draw polylines on marker for better visualization
            cv2.polylines(
                frame, [corner.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=5, lineType=cv2.LINE_AA
            )

            # Annotate Pose
            cv2.drawFrameAxes(frame, camMatrix, distCof, rVec[i], tVec[i], length=50, thickness=5)

            # # Draw a cross-mark at the centre of the frame (reference purposes only)
            # camera_center_x, camera_centre_y = camMatrix[0,2], camMatrix[1,2]
            # cv2.drawMarker(frame, (int(camera_center_x), int(camera_centre_y)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # result.write(frame)

    cv2.imshow("Pose Estimation Frame", frame)

    # Terminate program and cleanup when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

vs.release()
# result.release()

cv2.destroyAllWindows()
vs.stop()
