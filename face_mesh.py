import numpy as np
import cv2
import csv
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=5)

# def face_draw(image:cv2.Mat, text: str, focusText: str, p1: tuple(int, int), p2: tuple(int, int)):
#     cv2.putText(image, text, (p1[0]-50, p1[1]-150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(image, focusText, (p1[0]-50, p1[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.line(image, p1, p2, (0, 255, 0), 3)
#     # คำนวณตำแหน่งที่ต้องการแสดงข้อความ (ยกตัวอย่างว่าตำแหน่งนี้อยู่ด้านล่างกลางของ bounding box)
#     text_x = face_center_x
#     text_y = int(bounding_box_dict["xmin"] * img_h)  # แสดงที่ด้านบนของ bounding box
#     # คำนวณระยะห่างระหว่างจุดกึ่งกลางของ bounding box และจุดที่ต้องการแสดงข้อความ
#     distance_x = text_x - face_center_x
#     distance_y = text_y - face_center_y
#     # คำนวณระยะห่างรวม
#     distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
#     # คำนวณขนาดของตัวอักษร (ในที่นี้ใช้สูตรเพื่อปรับขนาดของตัวอักษรตามระยะห่าง)
#     font_scale = max(0.7, 2 - 0.1 * distance)
#     # แสดงข้อความบนภาพ
#     cv2.putText(image, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
#     return

def FaceMesh(image:cv2.Mat, face_imgs:[cv2.Mat], person_imgs: [cv2.Mat], bounding_boxs: [dict], focus_x:int, focus_y:int):
    if focus_x is None and focus_y is None:
        cv2.putText(image, "Please click to select focus point", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),  2,)
        return
    else:
        cv2.circle(image, (focus_x, focus_y), 10, (0, 0, 255), -1)

    for person_img, face_img, bounding_box_dict in zip(person_imgs, face_imgs, bounding_boxs):
        person_img_h, person_img_w, _ = person_img.shape

        face_center_x = int((bounding_box_dict["xmin"] + bounding_box_dict["width"] / 2) * person_img_w)
        face_center_y = int((bounding_box_dict['ymin'] + bounding_box_dict["height"] / 2) * person_img_h)

        # face mesh image size
        fm_img_h, fm_img_w, _ = face_img.shape

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

                cam_matrix = np.array([[focal_length, 0, person_img_w/2],
                                    [0, focal_length, person_img_h/2],
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

                p1 = (int(nose_2d[0] + bounding_box_dict["xmin"] * person_img_w),
                    int(nose_2d[1] + bounding_box_dict["ymin"] * person_img_h))
                p2 = (int(nose_2d[0] + (bounding_box_dict["xmin"] * person_img_w) + y * 2),
                    int(nose_2d[1] + (bounding_box_dict["ymin"] * person_img_h) - x * 2))

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
                    focusText = "go in to focus"
                else:
                    direct_in = 0
                    focusText = "go out of focus"
                cv2.line(person_img, p1, p2, (0, 255, 0), 3)
                # คำนวณตำแหน่งที่ต้องการแสดงข้อความ (ยกตัวอย่างว่าตำแหน่งนี้อยู่ด้านล่างกลางของ bounding box)
                text_x = face_center_x
                text_y = int(bounding_box_dict["ymin"] * person_img_h)  # แสดงที่ด้านบนของ bounding box
                # คำนวณระยะห่างระหว่างจุดกึ่งกลางของ bounding box และจุดที่ต้องการแสดงข้อความ
                distance_x = text_x - face_center_x
                distance_y = text_y - face_center_y
                # คำนวณระยะห่างรวม
                distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
                # คำนวณขนาดของตัวอักษร (ในที่นี้ใช้สูตรเพื่อปรับขนาดของตัวอักษรตามระยะห่าง)
                font_scale = max(0.7, 2 - 0.1 * distance)
                # แสดงข้อความบนภาพ
                cv2.putText(person_img, text, (text_x, text_y-60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                cv2.putText(person_img, focusText, (text_x, text_y-30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                cv2.putText(person_img, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)


                # row_list = ["Distance from focus", "slope", "face direct"]
                # d = round(d, 2)
                # row = [d, m, direct_in]
                # with open('data.csv', 'a+', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow(row)

                # input d , direct_in, m in ML output look/not-look
                # conclusion
                # look = ML(d,direct_in,m)
                # print(f"Distance from focus: {d}")