# -*- coding: utf-8 -*-
"""
Название программы: 
«Программа по определению амплитуды движений в основных суставах спортсмена с использованием искусственного интеллекта»
Авторы: 
1. Ермаков Алексей Валерьевич 
2. Белов Александр Генадьевич
3. Новосёлов Михаил Алексеевич
4. Скаржинская Елена Николаевна


Created on Thu Sep 23 20:08:16 2021

@author: Ермаков А.
"""

# Подключение библиотек
import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
####################################################
# Объявление переменных

stTime = time.time()
velLelbow2 = 0
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

# Установка размера шрифта и толщины линий
fontSize = 1
thick_line = 2


#############################################

# Именование процедур 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_pose = mp.solutions.pose
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)





# Загрузка видео
# Использование вэбкамеры
cap = cv2.VideoCapture(0)                          

# Загрузка заранее записанного файла
#cap = cv2.VideoCapture('F:/2.mp4')

p_time = 0


######################################
# Запись видео

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

# Объявление кодека и запись итогового видеофайла
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('F:/output.avi', fourcc, 20.0, size)


########################################

# Обработка видео
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()

        # Переход к цветовой схеме RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Определение позы
        results = pose.process(image)
        hresults = holistic.process(image)
        

        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, channels = image.shape
       
        
        
        
       
        
        
        # Вычисление углов сгиба
       
               
        def calculate_angle(a, b, c)-> None:
            a = np.array(a) # первая точка
            b = np.array(b) # общая точка
            c = np.array(c) # вторая точка

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
                
            
        def calculateNatAngle(a, b, c)-> None:
            a = np.array(a) # первая точка
            b = np.array(b) # общая точка
            c = np.array(c) # вторая точка

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angle = np.degrees(angle)
       
            

            return angle
            
        
##################################################################       
        

            
        # Определение координат точек
        try:
            landmarks = results.pose_landmarks.landmark
            wlandmarks = results.pose_world_landmarks.landmark
            
            

            

            # Координаты суставов

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
            
            
            
            # Вычисление углов в суставах
                

            lElbowAngle = (180 - int(calculateNatAngle(lshoulder, lelbow, lwrist)))
            rElbowAngle = (180 - int(calculateNatAngle(rshoulder, relbow, rwrist)))
            lShoulderAngle = (int(calculateNatAngle(lhip, lshoulder, lelbow)))
            rShoulderAngle = (int(calculateNatAngle(rhip, rshoulder, relbow)))
            rKneeAngle = (180 - int(calculateNatAngle(rhip, rknee, rankle)))
            lKneeAngle = (180 - int(calculateNatAngle(lhip, lknee, lankle)))
            rHipAngle = (180 - int(calculateNatAngle(rshoulder, rhip, rknee)))
            lHipAngle = (180 - int(calculateNatAngle(lshoulder, lhip, lknee)))
            lAnkleAngle = (180 - int(calculateNatAngle(lknee, lankle, lhill)))
            rAnkleAngle = (180 - int(calculateNatAngle(rknee, rankle, rhill)))
            lWristAngle = (180 - int(calculateNatAngle(lelbow, lwrist, lindex)))
            rWristAngle = (180 - int(calculateNatAngle(relbow, rwrist, rindex)))
            
           


        except:
            pass


        
####################################################################        
        
        

        # Нанесение изображения на видео

        
        # 1. Нанесение изображения на лицо
        mp_drawing.draw_landmarks(image, hresults.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=1))
        

        # 2. Нанесение изображения на левую кисть
        mp_drawing.draw_landmarks(image, hresults.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1))

        # 3. Нанесение изображения на правую кисть
        mp_drawing.draw_landmarks(image, hresults.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1))
        
        
        # 4. Нанесение изображения на вё тело
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(125,167,71), thickness=2, circle_radius=2))




        # Вычисление FPS
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        deltaTime = c_time-p_time
        p_time = c_time
        
        
        


        # Нанесение значения FPS на видео
        
        cv2.rectangle(image, (40, 0), (160, 70), (0,0,0), -1)
        cv2.putText(image, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        

        
        
        



        # Нанесение значений углов на видео
        

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


        
        

        # Нормализация видео
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resize(image, (640, 480))
        cv2.imshow('Image', image)



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        
        
        
        # Создание спсиков значений
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
        
        # Формирование таблицы значений
        a = {'Левый локоть': SpleftElbow, 'Правый локоть': SprightElbow,\
             'Левое плечо': SpleftShould, 'Правое плечо': SprightShould,\
             'Левое бедро': SpleftHip, 'Правое бедро': SprightHip,\
             'Левое колено': SpleftKnee, 'Правое колено': SprightKnee,\
             'Левый голеностоп': SpleftAnkle, 'Правый голеностоп': SprightAnkle,\
             'Левое запястье': SpleftWrist, 'Правое запястье': SprightWrist
            }
            
        df = pd.DataFrame(data=a)
              
        
        # Вывод значений в файл excel
        
        df.to_excel('F:/output.xlsx', sheet_name='Основные углы в проекции XY', index=False)
        
        
        
        # Передача видео к записи
        out.write(image) #output.avi 
        
        
        
        


# Завершение действия скрипта
cap.release()
out.release()
cv2.destroyAllWindows()

