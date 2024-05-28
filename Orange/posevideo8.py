# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:08:16 2021

@author: 1
"""

# Libraries
import mediapipe as mp
import cv2
#import matplotlib as plt
import time
import numpy as np
#import matplotlib.pyplot as plt
#from itertools import count


import math
import pandas as pd
import vg
import datetime
#from datetime import datetime

###############################################################################################################


#LeftElbowList = []# list for all points
counter1 = 0 # counter for printing
stTime = time.time()
stnTime=time.time_ns()

velLelbow2 = 0
df = pd.DataFrame(columns=['A', 'B']) # output table
#d = {'A': [], 'B': []}
#df = pd.DataFrame(data=d)
SpleftShould = []
SprightShould = []
SpleftElbow = []
SprightElbow = []
SpleftHip = []
SprightHip = []
SpleftKnee = []
SprightKnee = []

SpleftAnkle = []
SprightAnkle = []
SpleftWrist = []
SprightWrist = []

NSpleftShould = []
NSprightShould = []
NSpleftElbow = []
NSprightElbow = []
NSpleftHip = []
NSprightHip = []
NSpleftKnee = []
NSprightKnee = []

NSpleftAnkle = []
NSprightAnkle = []
NSpleftWrist = []
NSprightWrist = []


ttime=[]
ntime=[]
diftime=[]
x_fac = []

fontSize = 1
thick_line = 2


countFrame = 0
countFrameEror = 0
######################################################################################################

# DEFINITIONS 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_pose = mp.solutions.pose
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)




# FEED CAPTURE
#cap = cv2.VideoCapture(0)                           # Webcam feed
#cap = cv2.VideoCapture('G:/Dota/хфактор/2.mp4')
cap = cv2.VideoCapture('G:/Dota/хфактор/2.mov')
#cap = cv2.VideoCapture('files.upload()')
#cap = cv2.VideoCapture('G:/3.mp4')
#cap = cv2.VideoCapture("G:/2.mov")
p_time = 0


######################################
#Video saving

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('G:/Dota/хфактор/output.avi', fourcc, 20.0, size)


########################################






# DEFAULT COLOR VALUES
green = (0,255,0)
red = (0,0,255)
white = (255,255,255)
color1 = (0,255,0)
color2 = (0,255,0)
backColor = (255,155,155)



# INITIALIZATION
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Getting feed
    while cap.isOpened():
        ret, frame = cap.read()
        
        ##################################
        if not ret: 
            print("empty frame")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            #exit(1)
        #################################

        # Recolor formatting
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = pose.process(image)
        hresults = holistic.process(image)
        

        # Recolor format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, channels = image.shape
       
        
        '''
        ## grid linien optional
        xwd = (width//2)
        cv2.line(image, (xwd, 0), (xwd, height), (0,255,0), 2) # vertical line for orientation
        cv2.line(image, (0, height//2), (width, height//2), (0,255,0), 2) 
        '''
        
       
        
        
        # ANGLE CALCULATIONS
       
        
        # X-Y PLANE
        def calculate_angle(a, b, c)-> None:
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
        # Y-Z PLANE
        def calculate_angle_yz(a, b, c)-> None:
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[2]-b[2], c[1]-b[1]) - np.arctan2(a[2]-b[2], a[1]-b[1])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
        # X-Z PLANE
        def calculate_angle_xz(a, b, c)-> None:
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[2]-b[2], c[0]-b[0]) - np.arctan2(a[2]-b[2], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
       
################################################################################################################################################################        
       #NATURE ANGLE
        def NatureAngle(mainPoint, fPoint, sPoint):
            
            fVector = (fPoint[0]-mainPoint[0], fPoint[1]-mainPoint[1], fPoint[2]-mainPoint[2])
            sVector = (sPoint[0]-mainPoint[0], sPoint[1]-mainPoint[1], sPoint[2]-mainPoint[2])
            
            VecScal = (fVector[0]*sVector[0] + fVector[1]*sVector[1] + fVector[2]*sVector[2])
            VectAbs = ((fVector[0]**2 + fVector[1]**2 + fVector[2]**2)**0.5) * ((sVector[0]**2 + sVector[1]**2 + sVector[2]**2)**0.5)
            
            VecCos = VecScal / VectAbs
            angle = (math.acos(VecCos)*180.0)/np.pi                                    
            
            return angle
            
        def calculateNatAngle(a, b, c)-> None:
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angle = np.degrees(angle)
       
            '''
            if angle>180:
                angle = 360 - angle
            '''

            return angle
            
        
##############################################################################################################################################################        
        

            
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            wlandmarks = results.pose_world_landmarks.landmark
            #print(landmarks)
            

            

            # JOINTS

            lshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            lelbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z
            lwrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z
            rshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            relbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
            rwrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
            lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
            rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
            lknee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
            rknee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
            rankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
            lankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
            
            nose1 = landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y, landmarks[mp_pose.PoseLandmark.NOSE.value].z
            rhill = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z
            lhill = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z
            
            lindex = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z
            rindex = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z
            
            
            
            # ANGLE CALCULATIONS
            
            #XY
            

            lElbowAngle = (180 - int(calculate_angle(lshoulder, lelbow, lwrist)))
            rElbowAngle = (180 - int(calculate_angle(rshoulder, relbow, rwrist)))
            lShoulderAngle = (int(calculate_angle(lhip, lshoulder, lelbow)))
            rShoulderAngle = (int(calculate_angle(rhip, rshoulder, relbow)))
            rKneeAngle = (180 - int(calculate_angle(rhip, rknee, rankle)))
            lKneeAngle = (180 - int(calculate_angle(lhip, lknee, lankle)))
            rHipAngle = (180 - int(calculate_angle(rshoulder, rhip, rknee)))
            lHipAngle = (180 - int(calculate_angle(lshoulder, lhip, lknee)))
            
            lHeadAngle = (int(calculate_angle(nose1, lshoulder, rshoulder)))
            rHeadAngle = (179 - int(calculate_angle(nose1, rshoulder, lshoulder)))
            lAnkleAngle = (180 - int(calculate_angle(lknee, lankle, lhill)))
            rAnkleAngle = (180 - int(calculate_angle(rknee, rankle, rhill)))
            lWristAngle = (180 - int(calculate_angle(lelbow, lwrist, lindex)))
            rWristAngle = (180 - int(calculate_angle(relbow, rwrist, rindex)))
            
            # Z-AXIS ANGLES
            
            #XZ
            
            lElbowAngleXZ = (int(calculate_angle_xz(lshoulder, lelbow, lwrist)))
            rElbowAngleXZ = (int(calculate_angle_xz(rshoulder, relbow, rwrist)))
            lShoulderAngleXZ = (int(calculate_angle_xz(lhip, lshoulder, lelbow)))
            rShoulderAngleXZ = (int(calculate_angle_xz(rhip, rshoulder, relbow)))
            
            rKneeAngleXZ = (int(calculate_angle_xz(rhip, rknee, rankle)))
            lKneeAngleXZ = (int(calculate_angle_xz(lhip, lknee, lankle)))
            rHipAngleXZ = (int(calculate_angle_xz(rshoulder, rhip, rknee)))
            lHipAngleXZ = (int(calculate_angle_xz(lshoulder, lhip, lknee)))
            lHeadAngleXZ = (int(calculate_angle_xz(nose1, lshoulder, rshoulder)))
            rHeadAngleXZ = (int(calculate_angle_xz(nose1, rshoulder, lshoulder)))
            lAnkleAngleXZ = (180 - int(calculate_angle_xz(lknee, lankle, lhill)))
            rAnkleAngleXZ = (180 - int(calculate_angle_xz(rknee, rankle, rhill)))
            
            
            
            #YZ
            
            

            lElbowAngleYZ = (int(calculate_angle_yz(lshoulder, lelbow, lwrist)))
            rElbowAngleYZ = (int(calculate_angle_yz(rshoulder, relbow, rwrist)))
            lShoulderAngleYZ = (int(calculate_angle_yz(lhip, lshoulder, lelbow)))
            rShoulderAngleYZ = (int(calculate_angle_yz(rhip, rshoulder, relbow)))
            
            rKneeAngleYZ = (int(calculate_angle_yz(rhip, rknee, rankle)))
            lKneeAngleYZ = (int(calculate_angle_yz(lhip, lknee, lankle)))
            rHipAngleYZ = (int(calculate_angle_yz(rshoulder, rhip, rknee)))
            lHipAngleYZ = (int(calculate_angle_yz(lshoulder, lhip, lknee)))
            lHeadAngleYZ = (int(calculate_angle_yz(nose1, lshoulder, rshoulder)))
            rHeadAngleYZ = (int(calculate_angle_yz(nose1, rshoulder, lshoulder)))
            lAnkleAngleYZ = (180 - int(calculate_angle_yz(lknee, lankle, lhill)))
            rAnkleAngleYZ = (180 - int(calculate_angle_yz(rknee, rankle, rhill)))
            
            
            
            
            
 ############################################################################################################################################################# 

            
            #NATURE ANGLES#
            """
            lElbowShould = (lshoulder[0]-lelbow[0], lshoulder[1]-lelbow[1], lshoulder[2]-lelbow[2])
            lElbowWrist = (lwrist[0]-lelbow[0], lwrist[1]-lelbow[1], lwrist[2]-lelbow[2])
            lElbowScal = (lElbowShould[0]*lElbowWrist[0] + lElbowShould[1]*lElbowWrist[1] + lElbowShould[2]*lElbowWrist[2])
            lElbowAbs = ((lElbowShould[0]**2 + lElbowShould[1]**2 + lElbowShould[2]**2)**0.5) * (lElbowWrist[0]**2 + lElbowWrist[1]**2 + lElbowWrist[2]**2)**0.5
            lElbowCos = lElbowScal/lElbowAbs
            lElbowNature = (math.acos(lElbowCos)*180.0)/np.pi
            """  
            lElbowNature = 180 - round(NatureAngle(lelbow, lwrist, lshoulder))
            rElbowNature = 180 -round(NatureAngle(relbow, rwrist, rshoulder))
            lShoulderNature = round(NatureAngle(lshoulder, lhip, lelbow))
            rShoulderNature = round(NatureAngle(rshoulder, rhip, relbow))
            rKneeNature = 180 -round(NatureAngle(rknee, rhip, rankle))
            lKneeNature = 180 -round(NatureAngle(lknee, lhip, lankle))
            rHipNature = 180 -round(NatureAngle(rhip, rshoulder, rknee))
            lHipNature = 180 -round(NatureAngle(lhip, lshoulder, lknee))
            lAnkleNature = 180 -round(NatureAngle(lankle, lknee, lhill))
            rAnkleNature = 180 -round(NatureAngle(rankle, rknee, rhill))
            lWristNature = 180 -round(NatureAngle(lwrist, lelbow, lindex))
            rWristNature = 180 -round(NatureAngle(rwrist, relbow, rindex))
            
            
            lElbowNature1 = (int(calculateNatAngle(lshoulder, lelbow, lwrist)))
           




           
           


        except:
            print(f'Frame Error = {countFrameEror}')
            pass


        

############################################################################################################################################################  
#############################################################################################################################################################
       
       
        
        
        '''
       
        
        '''
################################################################################################################################################################## ################################################################################################################################################################       
        
        

        
        
        
        counter1 += 1
        #'''
        print(f'{counter1},{lElbowAngle},{rElbowAngle},{lShoulderAngle},{rShoulderAngle},\
{lKneeAngle},{rKneeAngle},{lHipAngle},{rHipAngle},{lHeadAngle},{rHeadAngle},{lAnkleAngle},{rAnkleAngle},\
{lElbowAngleXZ},{rElbowAngleXZ},{lShoulderAngleXZ},{rShoulderAngleXZ},{lKneeAngleXZ},{rKneeAngleXZ},\
{lHipAngleXZ},{rHipAngleXZ},{lHeadAngleXZ},{rHeadAngleXZ},{lAnkleAngleXZ},{rAnkleAngleXZ},\
{lElbowAngleYZ},{rElbowAngleYZ},{lShoulderAngleYZ},{rShoulderAngleYZ},{lKneeAngleYZ},{rKneeAngleYZ},\
{lHipAngleYZ},{rHipAngleYZ},{lHeadAngleYZ},{rHeadAngleYZ},{rAnkleAngleYZ},{rAnkleAngleYZ},\
{lElbowNature},{rElbowNature},{lShoulderNature},{rShoulderNature},{lKneeNature},{rKneeNature},\
{lHipNature},{rHipNature},{lAnkleNature},{rAnkleNature},{lWristNature},{rWristNature}')
        #'''
        '''
        for id, lm in enumerate(results.pose_landmarks.landmark):
            
            print(id, lm)
        '''
##############################################################################################################################################################        
        
        

        # LANDMARK DRAWINGS

        
        # 1. Face landmark drawing
        mp_drawing.draw_landmarks(image, hresults.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=1))
        

        # 2. Right Hand
        mp_drawing.draw_landmarks(image, hresults.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1))

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, hresults.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1))
        
        
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(125,167,71), thickness=2, circle_radius=2))




        # CALCULATE FPS
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        deltaTime = c_time-p_time
        p_time = c_time
        countFrame +=1
        
        
        #VELOCITY
        '''
        velLelbow1 = lElbowAngle*0.0174533
        #deltaVelelbow = round(abs(velLelbow1 - velLelbow2)*1000)
        deltaVelelbow = (velLelbow1 - velLelbow2)
        velLelbow2 = lElbowAngle*0.0174533
        '''
        velLelbow1 = lElbowAngle
        deltaVelelbow = abs(velLelbow1 - velLelbow2)
        velLelbow2 = lElbowAngle
        
        velElbGrad = round(deltaVelelbow/deltaTime)
        
        print('Velocity = ', velElbGrad, 'gr == ', deltaTime, 'time = ', time.time())
        '''
        #display velocity
        cv2.putText(image, str(velElbGrad),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+500, height+350]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        '''

        # DISPLAY FPS
        #"""
        cv2.rectangle(image, (40, 0), (120, 70), (0,0,0), -1)
        cv2.putText(image, str(int(fps)),  (40,25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(image,  str(countFrame), (40,60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        #"""

        
        
        



        # DISPLAY ANGLES
        
        #fontSize = 0.5
        
        #X-Y
        cv2.putText(image, str(lElbowAngle),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rElbowAngle),
                    tuple(np.multiply([relbow[0],relbow[1]], [width-150, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lShoulderAngle),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rShoulderAngle),
                    tuple(np.multiply([rshoulder[0], rshoulder[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lHipAngle),
                    tuple(np.multiply([lhip[0], lhip[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rHipAngle),
                    tuple(np.multiply([rhip[0], rhip[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lKneeAngle),
                    tuple(np.multiply([lknee[0], lknee[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rKneeAngle),
                    tuple(np.multiply([rknee[0], rknee[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        
        
        """
        cv2.putText(image, str(lElbowNature),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lElbowAngle),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lElbowAngleXZ),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(lElbowAngleYZ),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #"""
        
        '''
        cv2.putText(image, str((lelbow[2])),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        
        cv2.putText(image, str(rElbowNature),
                    tuple(np.multiply([relbow[0],relbow[1]], [width-150, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        
        cv2.putText(image, str(rElbowAngle),
                    tuple(np.multiply([relbow[0],relbow[1]], [width-150, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str((lwrist[2])),
                    tuple(np.multiply([lwrist[0], lwrist[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        cv2.putText(image, str(lShoulderNature),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        """
        cv2.putText(image, str(lShoulderAngle),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(lShoulderAngleXZ),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(lShoulderAngleYZ),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width+50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(lshoulder[2]),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        """
        
        
        #
        cv2.putText(image, str(rShoulderNature),
                    tuple(np.multiply([rshoulder[0], rshoulder[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        
        """
        cv2.putText(image, str(rShoulderAngle),
                    tuple(np.multiply([rshoulder[0], rshoulder[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        """
        
        # Hip
        
        cv2.putText(image, str(lHipNature),
                    tuple(np.multiply([lhip[0], lhip[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rHipNature),
                    tuple(np.multiply([rhip[0], rhip[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
                        
        
        
        cv2.putText(image, str(lHipAngle),
                    tuple(np.multiply([lhip[0], lhip[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rHipAngle),
                    tuple(np.multiply([rhip[0], rhip[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        
       
        
       # Knees 
        
        cv2.putText(image, str(lKneeNature),
                    tuple(np.multiply([lknee[0], lknee[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rKneeNature),
                    tuple(np.multiply([rknee[0], rknee[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        
        
        
        cv2.putText(image, str(lKneeAngle),
                    tuple(np.multiply([lknee[0], lknee[1]], [width+100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,100), thick_line, cv2.LINE_AA)
        
        cv2.putText(image, str(rKneeAngle),
                    tuple(np.multiply([rknee[0], rknee[1]], [width-100, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontSize, (100,255,255), thick_line, cv2.LINE_AA)
        '''


        
        

        # WINDOW NORMALISER
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resize(image, (640, 480))
        cv2.imshow('Image', image)



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        
        
        
        ### outputed in form in XY
        SpleftShould.append(lShoulderAngle)
        SprightShould.append(rShoulderAngle)
        
        SpleftElbow.append(lElbowAngle)
        SprightElbow.append(rElbowAngle)
        
        SpleftHip.append(lHipAngle)
        SprightHip.append(rHipAngle)
        
        SpleftKnee.append(lKneeAngle)
        SprightKnee.append(rKneeAngle)
        
        SpleftAnkle.append(lAnkleAngle)
        SprightAnkle.append(rAnkleAngle)
        
        SpleftWrist.append(lWristAngle)
        SprightWrist.append(rWristAngle)
        
        ### outputed in form in Nature
        
        NSpleftShould.append(lShoulderNature)
        NSprightShould.append(rShoulderNature)
        
        NSpleftElbow.append(lElbowNature)
        NSprightElbow.append(rElbowNature)
        
        NSpleftHip.append(lHipNature)
        NSprightHip.append(rHipNature)
        
        NSpleftKnee.append(lKneeNature)
        NSprightKnee.append(rKneeNature)
        
        NSpleftAnkle.append(lAnkleNature)
        NSprightAnkle.append(rAnkleNature)
        
        NSpleftWrist.append(lWristNature)
        NSprightWrist.append(rWristNature)
        
        #dt = datetime.datetime.now()
        #print('time =  ', dt.microsecond)
        
        
        def TimestampMillisec64():
            return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)
        
        
        
        ttime.append(time.time())
        ntime.append(time.time_ns())
        diftime.append(time.time_ns()-stnTime)
        
        #print('plechiii = ', np.array(lshoulder)-np.array(rshoulder) ) #rshoulder
        x_fac.append(vg.angle(np.array(lshoulder)-np.array(rshoulder), np.array(lhip)-np.array(rhip)))
        
       
        
        
        ### outputed dataset
        a = {'2D_Левый локоть': SpleftElbow, '2D_Правый локоть': SprightElbow,\
             '2D_Левое плечо': SpleftShould, '2D_Правое плечо': SprightShould,\
             '2D_Левое бедро': SpleftHip, '2D_Правое бедро': SprightHip,\
             '2D_Левое колено': SpleftKnee, '2D_Правое колено': SprightKnee,\
             '2D_Левый голеностоп': SpleftAnkle, '2D_Правый голеностоп': SprightAnkle,\
             '2D_Левое запястье': SpleftWrist, '2D_Правое запястье': SprightWrist,\
             '3D_Левый локоть': NSpleftElbow, '3D_Правый локоть': NSprightElbow,\
             '3D_Левое плечо': NSpleftShould, '3D_Правое плечо': NSprightShould,\
             '3D_Левое бедро': NSpleftHip, '3D_Правое бедро': NSprightHip,\
             '3D_Левое колено': NSpleftKnee, '3D_Правое колено': NSprightKnee,\
             '3D_Левый голеностоп': NSpleftAnkle, '3D_Правый голеностоп': NSprightAnkle,\
             '3D_Левое запястье': NSpleftWrist, '3D_Правое запястье': NSprightWrist,\
             'X-angle': x_fac, 'Time': ttime   
            }
        
        
            
        cs = {'leftElbow': NSpleftElbow, 'rightElbow': NSprightElbow,\
             'leftShould': NSpleftShould, 'rightShould': NSprightShould,\
             'leftHip': NSpleftHip, 'rightHip': NSprightHip,\
             'leftKnee': NSpleftKnee, 'rightKnee': NSprightKnee,\
             'leftAnkle': NSpleftAnkle, 'rightAnkle': NSprightAnkle,\
             'leftWrist': NSpleftWrist, 'rightWrist': NSprightWrist,\
             'Xangle': x_fac, 'time': ttime, 'nansec': ntime, 'diftime': diftime   
            }    
            
        #b = {'X-angle': x_fac, 'Time': ttime}    
        b = {'X-angle': x_fac}   
            
        df = pd.DataFrame(data=a)
        df_x = pd.DataFrame(data=b)
        df_cs = pd.DataFrame(data=cs)
              
        
        #### output in excel
        #df.to_excel('F:/1.xlsx', sheet_name='Основные углы в проекции XY', index=False)
        
        df.to_excel('G:/Dota/хфактор/output.xlsx', sheet_name='Основные углы', index=False)
        #df_cs.to_csv('G:/Dota/хфактор/output.csv',  index=False)
        #df_x.to_excel('G:/Dota/хфактор/X_factor.xlsx',  index=False)
        
        '''
        df1 = pd.DataFrame(data=a) 
        df2 = df1.copy()
        with pd.ExcelWriter('F:/1.xlsx') as writer:  
            df1.to_excel(writer, sheet_name='Sheet_name_1')
            df2.to_excel(writer, sheet_name='Sheet_name_2')
        '''

        
        
        
        
        
        
        
        
        
        
        # Saves for video
        out.write(image) #output.avi 
        
        
        
        

# When everything done, release the capture
#Deinitialize
cap.release()
out.release()
cv2.destroyAllWindows()

'''


'''
#plt.plot(SpleftElbow)
