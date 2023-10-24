import cv2
import numpy as np
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=5) # max_num_faces=5
mp_drawing = mp.solutions.drawing_utils

draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0)) # 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = face_mesh.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)

    cv2.imshow('MediaPipe FaceMesh', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break