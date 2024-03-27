import mediapipe as mp
import cv2


# FACIAL_KEYPOINTS = mp.solutions.face_detection.FaceKeyPoint
# print(FACIAL_KEYPOINTS)
right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_mesh
mp_face_mesh = mp_face_detection.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = mp_face_mesh.process(img_rgb)
    if results.multi_face_landmarks:    
        right_eye_list = []
        for facial_landmarks in results.multi_face_landmarks:
            h, w, c = img.shape
            # eye_left=mp_face_detection.get_key_point(facial_landmarks, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            '''
            pt0 = facial_landmarks.left_eye_boundary_landmarks
            h, w, c = img.shape
            cx, cy = int(i.x * w), int(i.y * h)
            cv2.circle(img, (cx, cy), 1, (0, 0, 255), cv2.FILLED)'''
            for i in right_eye_landmarks:
                temp_pt = facial_landmarks.landmark[i]
                cx, cy = int(temp_pt.x * w), int(temp_pt.y * h)
                right_eye_list.append([cx, cy])
                cv2.circle(img, (cx, cy), 1, (0, 0, 255), cv2.FILLED)
            
            
                
            
            
            
            
    cv2.imshow("Face Window", img)
    
    if cv2.waitKey(1)==27:
        cv2.destroyAllWindows()
        cap.release()
        break
        
    