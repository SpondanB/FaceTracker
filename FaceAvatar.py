import pygame
import sys
import math

import mediapipe as mp
import cv2

right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
left_eye_landmarks = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
base_face_landmarks = [103, 67, 109, 10, 338,297, 332, 372, 352, 433, 421, 201, 213, 123, 143] # , 432, 212
# base_face_landmarks = [103, 105, 10, 334, 332, 372, 352, 433, 421, 201, 213, 123, 143] # , 432, 212
over_face_landmarks = [105, 334, 374, 330, 322, 92, 101, 145]
over_face_left_landmarks = [334, 332, 372, 352, 433, 432, 322, 330, 374]
over_face_right_landmarks = [212, 213, 123, 143, 103, 105, 145, 101, 92]
over_nose_left_landmarks = [6, 4, 305]
over_nose_right_landmarks = [6, 4, 75]
under_nose_landmarks = [4, 75, 305]
mouth_landmarks = [80, 81, 82, 13, 312, 311, 310, 402, 317, 14, 87, 178]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_mesh
mp_face_mesh = mp_face_detection.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
success , img = cap.read()

fps = 32
h, w, c = img.shape
screen = pygame.display.set_mode((w, h))
fps_clock = pygame.time.Clock()

while True:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            break
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = mp_face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks: 
        screen.fill((0, 255, 0))
        
        # required points
        left_eye_list = []
        right_eye_list = []
        base_face_list = []
        over_face_list = []
        over_face_left_list = []
        over_face_right_list = []
        under_nose_list = []
        over_nose_left_list = []
        over_nose_right_list = []
        mouth_list = []
        
        # getting the points
        for facial_landmarks in results.multi_face_landmarks:
            for i in right_eye_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                right_eye_list.append([cx, cy])
            for i in left_eye_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                left_eye_list.append([cx, cy])
                # cv2.circle(img, (cx, cy), 1, (0, 0, 255), cv2.FILLED)
            for i in base_face_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                base_face_list.append([cx, cy])
            for i in over_face_left_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                over_face_left_list.append([cx, cy])
            for i in over_face_right_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                over_face_right_list.append([cx, cy])
            for i in over_face_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                over_face_list.append([cx, cy])
            for i in under_nose_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                under_nose_list.append([cx, cy])
            for i in over_nose_left_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                over_nose_left_list.append([cx, cy])
            for i in over_nose_right_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                over_nose_right_list.append([cx, cy])
            '''
            for i in mouth_landmarks:
               temp_pt = facial_landmarks.landmark[i]
               cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
               mouth_list.append([cx, cy])
            '''
            tp1 = facial_landmarks.landmark[81]
            tp2 = facial_landmarks.landmark[82]
            tp3 = facial_landmarks.landmark[87]
            x1, y1 = int(((tp1.x+tp2.x+tp3.x)/3)*w), int(((tp1.y+tp2.y+tp3.y)/3)*h) 
            mouth_list.append([x1, y1])
            temp_pt = facial_landmarks.landmark[13]
            cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
            mouth_list.append([cx, cy])
            tp1 = facial_landmarks.landmark[311]
            tp2 = facial_landmarks.landmark[312]
            tp3 = facial_landmarks.landmark[317]
            x1, y1 = int(((tp1.x+tp2.x+tp3.x)/3)*w), int(((tp1.y+tp2.y+tp3.y)/3)*h) 
            mouth_list.append([x1, y1])
            temp_pt = facial_landmarks.landmark[14]
            x1, y1 = int(temp_pt.x * w), int(temp_pt.y * h)
            if math.hypot(cx-x1, cy-y1) <= 2:
                x1 += 2
                y1 += 2
            mouth_list.append([x1, y1])
            
        # face print
        pygame.draw.polygon(screen, (0, 0, 0), base_face_list)
        pygame.draw.polygon(screen, (0, 0, 0), over_face_left_list)
        pygame.draw.polygon(screen, (0, 0, 0), over_face_right_list)
        pygame.draw.polygon(screen, (0, 0, 0), over_face_list)
        
        # nose print
        pygame.draw.polygon(screen, (0, 0, 255), over_nose_left_list)
        pygame.draw.polygon(screen, (0, 0, 255), over_nose_right_list)
        pygame.draw.polygon(screen, (50, 50, 50), under_nose_list)
        
        # mouth print
        pygame.draw.polygon(screen, (255, 0, 0), mouth_list)
        
        # eye print
        pygame.draw.polygon(screen, (255, 255, 255), right_eye_list)
        pygame.draw.polygon(screen, (255, 255, 255), left_eye_list)
        
        # update screen
        pygame.display.update()
        
    fps_clock.tick(fps)
            