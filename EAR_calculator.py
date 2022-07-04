# Define the function for calculating the Eye Aspect Ratio(EAR)
from scipy.spatial import distance as dist 
import numpy as np
import cv2
import math
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
	# Vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal eye landmarks 
	# compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	EAR = (A + B) / (2.0 * C)
	return EAR

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    MAR = (A + B + C) / (2.0 * D)
    return MAR

def cal_angle(a,b,c):
    v1 = np.array(a) - np.array(b)
    v0 = np.array(b) - np.array(c)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)
#  # Convert from rotation vector to Euler angle
# def get_euler_angle(rotation_vector):
#     # calculate rotation angles
#     theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
#     # transformed to quaterniond
#     w = math.cos(theta / 2)
#     x = math.sin(theta / 2)*rotation_vector[0][0] / theta
#     y = math.sin(theta / 2)*rotation_vector[1][0] / theta
#     z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
#     ysqr = y * y
#     # pitch (x-axis rotation)
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x * x + ysqr)
#     print('t0:{}, t1:{}'.format(t0, t1))
#     pitch = math.atan2(t0, t1)
    
#     # yaw (y-axis rotation)
#     t2 = 2.0 * (w * y - z * x)
#     if t2 > 1.0:
#         t2 = 1.0
#     if t2 < -1.0:
#         t2 = -1.0
#     yaw = math.asin(t2)
    
#     # roll (z-axis rotation)
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 - 2.0 * (ysqr + z * z)
#     roll = math.atan2(t3, t4)
    
#     print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
# 	 # Unit conversion: convert radians to degrees
#     Y = int((pitch/math.pi)*180)
#     X = int((yaw/math.pi)*180)
#     Z = int((roll/math.pi)*180)
    
#     return 0, Y, X, Z