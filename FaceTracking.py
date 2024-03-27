'''
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True , min_tracking_confidence=0.6 ,min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

ret, frm = cap.read()

while ret:
    ret, frm = cap.read()
    
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    
    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            draw.draw_landmarks(frm, i, facmesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
    
    cv2.imshow("Face Window", frm)
    
    if cv2.waitKey(1)==27:
        cv2.destroyAllWindows()
        cap.release()
        break
'''

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style()) # tesselation_mask
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())  # default facemesh of lines
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())  # draws only the iris
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        cap.release()
        break


