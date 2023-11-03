import cv2
import numpy as np
import mediapipe as mp
import time
import csv
import os
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=5)
mp_drawing = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

draw_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0))

focus_x, focus_y = None, None



def Book(image: cv2.Mat):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results_face_detect = face_detection.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    if results_face_detect.detections:
        for detection in results_face_detect.detections:
            # bounding box  
            xmin = detection.location_data.relative_bounding_box.xmin
            ymin = detection.location_data.relative_bounding_box.ymin
            width = detection.location_data.relative_bounding_box.width
            height = detection.location_data.relative_bounding_box.height
            
            if(xmin < 0) or (ymin < 0):
                continue
            
            cv2.rectangle(image,
              (int(xmin * img_w),  # กว้าง
               int(ymin * img_h)),  # สูง
              (int((xmin + width) * img_w),
               int((ymin + detection.location_data.relative_bounding_box.height) * img_h)),
              (255, 255, 255), 2)
            
            if focus_x is None and focus_y is None:
                cv2.putText(image, "Please select focus point", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                continue
            else:
                cv2.circle(image, (focus_x, focus_y), 10, (0, 0, 255), -1)
            # get face image
            face_img = image[int(ymin * img_h):int((ymin + height) * img_h),
                            int(xmin * img_w):int((xmin + width) * img_w)]
            # find center of face in large image
            face_center_x = int((xmin + width / 2) * img_w)
            face_center_y = int((ymin + height / 2) * img_h)

            # face mesh image size
            fm_img_h, fm_img_w, fm_img_c = face_img.shape

            # face mesh image center
            fm_focus_x = int(fm_img_w / 2)
            fm_focus_y = int(fm_img_h / 2)

            # face mesh
            results_face_mesh = face_mesh.process(face_img)

            # if face mesh detected
            if results_face_mesh.multi_face_landmarks:
                # loop for each face
                for i, face_landmarks in enumerate(results_face_mesh.multi_face_landmarks):
                    face_3d = []
                    face_2d = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 6:
                            if idx == 6:
                                # if idx == 33 or idx == 133 or idx == 159 or idx == 144 or idx == 153 or idx == 145:
                                #     if idx == 159 :
                                # if idx == 362 or idx == 263 or idx == 386 or idx == 380 or idx == 374 or idx == 373:
                                #     if idx == 386 :
                                nose_2d = (lm.x * fm_img_w, lm.y * fm_img_h)
                                nose_3d = (lm.x * fm_img_w, lm.y *
                                           fm_img_h, lm.z * 3000)
                            x, y = int(lm.x * fm_img_w), int(lm.y * fm_img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * fm_img_w

                    cam_matrix = np.array([[focal_length, 0, img_w/2],
                                           [0, focal_length, img_h/2],
                                           [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix)

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

                    nose_3d_projected, jac = cv2.projectPoints(
                        nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
    
                    # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    p1 = (int(nose_2d[0] + xmin * img_w),
                          int(nose_2d[1] + ymin * img_h))
                    p2 = (int(nose_2d[0] + (xmin * img_w) + y * 2),
                          int(nose_2d[1] + (ymin * img_h) - x * 2))

                    cv2.putText(
                        image, text, (p1[0]-50, p1[1]-150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if p2[0] - p1[0] == 0:
                        d = abs(p1[0]-fm_focus_x)
                    else:
                        m = -(p2[1] - p1[1]) / (p2[0] - p1[0])
                        b = p1[1] - m * p1[0]
                        d = abs(m * fm_focus_x - fm_focus_y + b) / \
                            (m ** 2 + 1) ** 0.5  # line with focus point

                    d1 = (((p1[0] - focus_x) ** 2) +
                          ((p1[1] - focus_y) ** 2)) ** 0.5  # focus-p1
                    d2 = (((p2[0] - focus_x) ** 2) +
                          ((p2[1] - focus_y) ** 2)) ** 0.5  # focus-p2
                    if (d2 < d1):
                        direct_in = 1
                        cv2.putText(
                            image, "go into focus", (p1[0]-50, p1[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        direct_in = 0
                        cv2.putText(image, "go out from focus", (
                            p1[0]-50, p1[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    row_list = ["Distance from focus", "slope", "face direct"]
                    d = round(d, 2)
                    row = [d, m, direct_in]
                    with open('data.csv', 'a+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

                    # input d , direct_in, m in ML output look/not-look
                    # conclusion
                    # look = ML(d,direct_in,m)
                    # print(f"Distance from focus: {d}")
                    cv2.line(image, p1, p2, (0, 255, 0), 3)

                    # คำนวณตำแหน่งที่ต้องการแสดงข้อความ (ยกตัวอย่างว่าตำแหน่งนี้อยู่ด้านล่างกลางของ bounding box)
                    text_x = face_center_x
                    text_y = int(ymin * img_h)  # แสดงที่ด้านบนของ bounding box

                    # คำนวณระยะห่างระหว่างจุดกึ่งกลางของ bounding box และจุดที่ต้องการแสดงข้อความ
                    distance_x = text_x - face_center_x
                    distance_y = text_y - face_center_y

                    # คำนวณระยะห่างรวม
                    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

                    # คำนวณขนาดของตัวอักษร (ในที่นี้ใช้สูตรเพื่อปรับขนาดของตัวอักษรตามระยะห่าง)
                    font_scale = max(0.7, 2 - 0.1 * distance)

                    # แสดงข้อความบนภาพ
                    cv2.putText(image, str(i), (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    else:
        cv2.putText(image, "No any face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MediaPipe FaceMesh', image)
    def draw_circle(event, x, y, flags, param):
        global focus_x, focus_y
        if event == cv2.EVENT_LBUTTONUP:
            focus_x, focus_y = x, y

    cv2.setMouseCallback('MediaPipe FaceMesh', draw_circle)
