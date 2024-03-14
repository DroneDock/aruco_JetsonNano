"""
CAMERA CALIBRATION PART (C)

This script performs camera calibration using ArUco markers on a board. Here are a few steps to follow prior to running this script:
    a) Preparing the resources for camera calibration as carried out in 'aruco_board_generation.py' and 'data_generation.py'.
        1) Print the aruco marker board of DICT_6x6_50
        2) Take around 50 images of the printed board pasted on a flat card-board, from different angles.
        3) Set path to store images first

    b) Calibrating Camera
        1) Measure length of the side of individual marker and spacing between two marker
        2) Input above data (length and spacing) in "camera_calibration.py"
        3) Set path to stored images of aruco marker

There are in total two functions of this code:
    a) Camera Calibration (if calibrate_camera is True):
        - The camera matrix and distortion coefficients will be calculated and stored in the YAML file.

    b) Real-time Validation (if calibrate_camera is False)
        - The real-time validation assumes a calibration has been performed and the calibration data is stored in calibration.yaml. 
        - The pose of the ArUco marker board is estimated, and the result is visualized with markers and coordinate axes.

Created by: Jalen
"""

import time
import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imutils.video import VideoStream

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = False

# Set path to the images
calib_imgs_path = root.joinpath("aruco_calibration_data")



# DEFINING ARUCO BOARD PARAMETERS ----------------------------------------------------------------------------------------------------------
# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_50 )

#Provide length of the marker's side
markerLength = 3.50  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)


'''uncomment following block to draw and show the board'''
# img = board.draw((864,1080))
# cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters_create()



# CAMERA CALIBRATION ----------------------------------------------------------------------------------------------------------
if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread( str(root.joinpath(fn) ))
        img_list.append( img )
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    # Save the camera matrix and distortion coefficients to a YAML file (calibration.yaml).
    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)




# REAL TIME VALIDATION ----------------------------------------------------------------------------------------------------------
# else:
#     # Open a video stream 
#     # camera = cv2.VideoCapture(0)

#     camera = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)500, height=(int)500,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink")
#     time.sleep(2)

#     ret, img = camera.read()

#     # Load the camera matrix and distortion coefficients from YAML file
#     with open('calibration.yaml') as f:
#         loadeddict = yaml.load(f, Loader=yaml.FullLoader)
#     mtx = loadeddict.get('camera_matrix')
#     dist = loadeddict.get('dist_coeff')
#     mtx = np.array(mtx)
#     dist = np.array(dist)

#     ret, img = camera.read()
#     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#     h,  w = img_gray.shape[:2]
#     newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#     last_print_time = time.time()

#     # Detect ArUco markers in undistorted frames
#     # Estimate the pose (rotation and translation) of the ArUco marker board
#     pose_r, pose_t = [], []
#     while True:
#         ret, img = camera.read()
#         img_aruco = img
#         im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         h,  w = im_gray.shape[:2]
#         dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
#         corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
#         #cv2.imshow("original", img_gray)
#         if corners == None:
#             print ("pass")
#         else:

#             # ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
#             ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist, rvec=None, tvec=None)
            
#             if ret != 0:
#                 img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
#                 img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

#                 # # Axis length (you can change it according to your requirement)
#                 # axis_length = 10

#                 # # Project 3D points to image plane
#                 # axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3, 1)
#                 # imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, newcameramtx, dist)

#                 # Draw axis lines on the image
#                 img_aruco = cv2.drawFrameAxes(img_aruco, newcameramtx, dist, rvec, tvec, 10)


#                 # Print relative distance values every 2 seconds
#                 current_time = time.time()
#                 if current_time - last_print_time >= 2.0:
#                     print("Rotation:", rvec.flatten())
#                     print("Translation:", tvec.flatten())
#                     print("-----------------------------")
#                     last_print_time = current_time

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break;
#         cv2.imshow("World co-ordinate frame axes", img_aruco)

# cv2.destroyAllWindows()





else:
    # Open a video stream 
    # camera = cv2.VideoCapture(0)

    camera = VideoStream("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)500, height=(int)500,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink").start()
    time.sleep(2)


    # Load the camera matrix and distortion coefficients from YAML file
    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)


    count = 0
    while True:
        # Read a frame from the camera
        frame = camera.read()

        # Check if the frame is not None
        if frame is not None:
            # Print the shape of the frame for debugging
            print(frame.shape)

            # Detect ArUco markers in undistorted frames
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = im_gray.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict)

            if corners is not None:
                # ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist)  # For a board
                ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist, rvec=None, tvec=None)

                if ret != 0:
                    frame = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
                    frame = aruco.drawAxis(frame, newcameramtx, dist, rvec, tvec, 10)  # axis length 100 can be changed

                    # Print relative distance values every 2 seconds
                    current_time = time.time()
                    if current_time - last_print_time >= 2.0:
                        print("Rotation:", rvec.flatten())
                        print("Translation:", tvec.flatten())
                        print("-----------------------------")
                        last_print_time = current_time

            # Display the current frame
            cv2.imshow("img", frame)

        # If 'q' key is pressed, exit the loop
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Release the camera (or stop the VideoStream) and close the OpenCV windows
    camera.stop()
    cv2.destroyAllWindows()