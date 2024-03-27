import mediapipe as mp
import cv2
import csv
# import socket
# import json

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Create a socket object
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and port
# host = '127.0.0.1'  # Localhost
# port = 12345

# Bind the socket to the host and port
# server_socket.bind((host, port))

# Start listening for incoming connections
# server_socket.listen(1)
# print(f"Server listening on {host}:{port}")

# Accept a connection from a client
# client_socket, client_address = server_socket.accept()
# print(f"Connection from {client_address} has been established.")

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# Open a CSV file for writing
with open('facial_landmarks.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Landmark Index', 'X', 'Y', 'Z'])  # Write header row

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        # Write landmark data to CSV file
                        csvwriter.writerow([idx, landmark.x, landmark.y, landmark.z])

                    # landmark_data = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
                    # Convert landmark data to JSON format
                    # landmark_json = json.dumps(landmark_data)
                    # Send landmark data over the socket
                    # client_socket.sendall(landmark_json.encode())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_NOSE,
                        landmark_drawing_spec=None)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None)

            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

            if cv2.waitKey(1) & 0xFF == 27:
                break

# Close the CSV file
csvfile.close()

# Close the client socket
# client_socket.close()

# Close the server socket
# server_socket.close()

# Destroy all the windows
cv2.destroyAllWindows()
