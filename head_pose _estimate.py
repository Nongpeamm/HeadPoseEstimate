import cv2
import numpy as np
import mediapipe as mp
import time
import csv 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=5)
mp_drawing = mp.solutions.drawing_utils

draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    
    start = time.time()
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    
    focus_x = int(img_w / 2)
    focus_y = int(img_h / 2)
    cv2.circle(image, (focus_x,focus_y), radius= 1, color=(0, 0, 255), thickness=5)
    
    if results.multi_face_landmarks:
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 6:
                    if idx == 6 :
                # if idx == 33 or idx == 133 or idx == 159 or idx == 144 or idx == 153 or idx == 145:
                #     if idx == 159 :
                # if idx == 362 or idx == 263 or idx == 386 or idx == 380 or idx == 374 or idx == 373:
                #     if idx == 386 :
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            focal_length = 1 * img_w
            
            cam_matrix = np.array([ [focal_length, 0, img_w/2],
                                    [0, focal_length, img_h/2],
                                    [0, 0, 1]])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rotation_vector)
            
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            if y < -10:
                text = 'looking left'
            elif y > 10:
                text = 'looking right'
            elif x < -10:
                text = 'looking down'
            elif x > 10:
                text = 'looking up'
            else:
                text = 'looking forward'
                
            nose_3d_projected, jac = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.putText(image, text, (p1[0]-50, p1[1]-150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if p2[0] - p1[0] == 0:
                d = abs(p1[0]-focus_x)
            else :
                m = -(p2[1] - p1[1]) / (p2[0] - p1[0])
                b = p1[1] - m * p1[0]
                d = abs(m * focus_x - focus_y + b) / (m ** 2 + 1) ** 0.5 # line with focus point
            
            d1 = (((p1[0] - focus_x) ** 2) + ((p1[1] - focus_y) ** 2)) ** 0.5  # focus-p1
            d2 = (((p2[0] - focus_x) ** 2) + ((p2[1] - focus_y) ** 2)) ** 0.5  # focus-p2 
            if(d2 < d1) :
                direct_in = 1
                cv2.putText(image, "go into focus", (p1[0]-50, p1[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                direct_in = 0
                cv2.putText(image, "go out from focus", (p1[0]-50, p1[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            row_list = ["Distance from focus", "slope", "face direct"]
            d = round(d, 2)
            row = [d, m, direct_in]
            with open('data.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            
            # input d , direct_in, m in ML output look/not-look
            #conclusion
            # look = ML(d,direct_in,m)
            print(f"Distance from focus: {d}")
            cv2.line(image, p1, p2, (0, 255, 0), 3)
            cv2.putText(image, str(i), (p1[0], p1[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        end = time.time()
        totalTime = end - start
        
        fps = 1 / totalTime
        # print("FPS: ", fps)
        
    else:
        cv2.putText(image, "No any face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('MediaPipe FaceMesh', image)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("ๆ"):
        break
        
cap.release()
cv2.destroyAllWindows()
    