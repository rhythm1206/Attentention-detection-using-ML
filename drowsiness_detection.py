# Import the necessary packages 
import csv
import datetime as dt
import os
import time
from datetime import datetime

import cv2
import dlib
import imutils
import matplotlib.animation as animation
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imutils import face_utils
from imutils.video import VideoStream
from matplotlib import style
from scipy.spatial import distance as dist

from EAR_calculator import *
from gaze_tracking import GazeTracking

style.use('fivethirtyeight')
# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


#all eye  and mouth aspect ratio with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer/trainer.yml')
# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
EAR_THRESHOLD = 0.3
# Declare another costant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 20
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 0.46

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

gaze = GazeTracking()
# Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) gets us the first and last coordinates for the mouth.
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO]Loading Camera.....")
vs = cv2.VideoCapture(0)
# time.sleep(1.0) 

assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0 
count_focus = 0
count_lossfocus = 0
# Now, loop over all the frames and detect the faces
while True: 
    # Extract a frame 
    _, frame = vs.read()
    frame  = cv2.flip(frame , 1)
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    # Resize the frame 
    frame = imutils.resize(frame, width=750)
    # Convert the frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    # rects = detector(frame, 1)
    rects= detector(gray,0)
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""
    if(len(rects)==0):
        cv2.putText(frame, "-----Face Not Detected-----", (85, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        # Now loop over all the face detections and apply the predictor 
        for (i, rect) in enumerate(rects): 
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(shape)

            # Draw a rectangle over the detected face 
            (x, y, w, h) = face_utils.rect_to_bb(rect) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            # attention_percent = ((len(shape) - total_del)*100)/(len(shape)+0.1)
            # _,confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Put a number 
            cv2.putText(frame, "Student", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # for headpose
            #2D image points. If you change the image, you need to change vector
            image_points = np.array([
                                        (shape[30, :]),     # Nose tip
                                        (shape[8,  :]),     # Chin
                                        (shape[36, :]),     # Left eye left corner
                                        (shape[45, :]),     # Right eye right corne
                                        (shape[48, :]),     # Left Mouth corner
                                        (shape[54, :])      # Right mouth corner
                                    ], dtype="double")

            # 3D model points.
            model_points = np.array([
                                        (0.0, 0.0, 0.0),             # Nose tip
                                        (0.0, -330.0, -65.0),        # Chin
                                        (-225.0, 170.0, -135.0),     # Left eye left corner
                                        (225.0, 170.0, -135.0),      # Right eye right corne
                                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                                        (150.0, -150.0, -125.0)      # Right mouth corner
                                    ])
            # Camera internals
            size = frame.shape
            focal_length = size[1]
            center = (size[1]//2, size[0]//2)
            camera_matrix = np.array(
                                     [[focal_length, 0, center[0]],
                                     [0, focal_length, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )
            #print ("Camera Matrix :\n {0}".format(camera_matrix))

            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            #print ("Rotation Vector:\n {0}".format(rotation_vector))
            #print ("Translation Vector:\n {0}".format(translation_vector))

            # Project a 3D point (0, 0, 1000.0) onto the frame.
            # We use this to draw a line sticking out of the nose
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            # x1, x2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix)
            p0 = ( int(image_points[0][0]), int(image_points[0][1]) - 2)
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            


            cv2.line(frame, p1, p2, (255,0,0), 2)
            # cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)
            angle = cal_angle(p0,p1,p2)

            # calc euler angle
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)


        

           
            #for eye
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend] 
            #for mouth
            mouth = shape[mstart:mend]

            # Compute the EAR for both the eyes 
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            #live datawrite in csv
            ear_list.append(EAR)
            #print(ear_list)


            ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # Compute the convex hull for both the eyes and then visualize it
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # Draw the contours 
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255,0), 1)
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

            MAR = mouth_aspect_ratio(mouth)
            mar_list.append(MAR/10)

            
            if gaze.is_center and (-10 <= euler_angle[2, 0] and euler_angle[2, 0] <= 10):
                    count_focus +=1
                    cv2.putText(frame, "ATTENTIVE!", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                count_lossfocus +=1
                cv2.putText(frame, "** LOST FOCUS! **", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite("dataset/lost_focus/lost_focus%d.jpg" % count_lossfocus, frame)


                # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
                # Thus, count the number of frames for which the eye remains closed 
                if EAR < EAR_THRESHOLD : 
                    FRAME_COUNT += 1

                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
                    
                    # Add the frame to the dataset ar a proof of drowsy driving
                    if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                            count_sleep += 1
                            cv2.putText(frame, "** DROWSINESS ALERT! **", (270, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # Add the frame to the dataset ar a proof of drowsy driving
                            cv2.imwrite("dataset/drowsy/frame_sleep%d.jpg" % count_sleep, frame)
                else:
                    FRAME_COUNT = 0
                    
                cv2.putText(frame, "EAR: {:.2f}".format(EAR), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if the person is yawning
                if MAR > MAR_THRESHOLD:
                    count_yawn += 1
                    cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
                    cv2.putText(frame, "YAWNING!", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Add the frame to the dataset ar a proof of drowsy driving
                    cv2.imwrite("dataset/yawn/frame_yawn%d.jpg" % count_yawn, frame)
                    # playsound('sound files/alarm.mp3')
                    # playsound('sound files/warning_yawn.mp3')
                
            
        #total data collection for plotting
        for i in ear_list:
            total_ear.append(i)
        for i in mar_list:
            total_mar.append(i)		
        for i in ts:
            total_ts.append(i)
    #display the frame 
    cv2.imshow("Output", frame)
    key = cv2.waitKey(50) & 0xff 
    
    if key == 27:
        exit()

    if key == ord('q'):
        break
    

a = total_ear
b = total_mar
# c = total_ang
d = total_ts
print(total_ang)
df = pd.DataFrame({"EAR" : a, "MAR":b, "TIME" : d})

df.to_csv("op_webcam.csv", index=False)
df=pd.read_csv("op_webcam.csv")

df.plot(x='TIME',y=['EAR','MAR'])
#plt.xticks(rotation=45, ha='right')

plt.subplots_adjust(bottom=0.30)
plt.title('EAR & MAR calculation over time of webcam')
plt.ylabel('EAR & MAR')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
cv2.destroyAllWindows()
vs.release()
