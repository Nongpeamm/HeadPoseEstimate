import os
import cv2
import requests
import torch
from threading import Thread
from boxmot import create_tracker, get_tracker_config
from dotenv import load_dotenv
from ultralytics import YOLO
from detect import DetectFasion, DetectFace
from pathlib import Path
from colors import Color
from load_model import YoloDetectWithTracker
from insightface.app import FaceAnalysis
import numpy as np

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

video_writer = cv2.VideoWriter(
        "test_beam.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

def detect_person_cam1(model: DetectFasion, image:cv2.Mat, id_dict: dict, DICT_MAX_SIZE: int):
    person_imgs, track_ids = model.person_detect(image)

    for person_img, track_id in zip(person_imgs, track_ids):
        if track_id not in id_dict:
            length = len(id_dict)
            if length > 0 and track_id < list(id_dict)[-1]:
                continue
            
            if length > DICT_MAX_SIZE:
                print("Sending data to server")
                Thread(target=requests.post, args=(os.getenv("API_URL") + "/customer",), kwargs={"json": id_dict, "headers": { "accessToken": token }}).start()
                id_dict.clear()

            id_dict[track_id] = {
                "dress": 0,
                "t-shirt": 0,
                "jacket": 0,
                "top": 0,
                "long-sleeve": 0,
                "short": 0,
                "skirt": 0,
                "trouser": 0
            }

        _, fashion_labels = model.fashion_detect(person_img)
        for fashion_label in fashion_labels:
            id_dict[track_id][fashion_label] += 1

        # print(id_dict)
        video_writer.write(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA))

def detect_face_cam2(person_detect_model: YOLO, image:cv2.Mat, face_detect_model: FaceAnalysis):
    
    # person_imgs, _ = YoloDetectWithTracker(
    #         image, person_detect_model, sort_tracker, conf=0.4)  # return person images
        # bbox_list, face_landmark = face_detect_model.detect(image)
        # if bbox_list is None or bbox_list.shape[0] == 0 or face_landmark is None:
        #     return

        # bbox = bbox_list[0][0:4]
        # x1, y1, x2, y2 = bbox
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(image, (x1, y1), (x2, y2), Color.purple, 2)

        # x, y = face_landmark[0][2]  # nose
        # # cv2.circle(person_img, (int(x), int(y)), 1, Color.red, 2) #error CUDA
        # cv2.rectangle(image, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), Color.red, 2)
        
    # init camera coordinate
    focal_length = image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    cv2. circle(image, (int(center[0]), int(center[1])), 6, Color.pink, 6)
    camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    resized_img = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    face_imgs = face_detect_model.get(resized_img)
    
    # face_img.landmark_3d_68[8] # chin
    # face_img.landmark_3d_68[30] # nose
    # face_img.landmark_3d_68[45] # left eye
    # face_img.landmark_3d_68[36] # right eye
    # face_img.landmark_3d_68[48] # left mouth
    # face_img.landmark_3d_68[54] # right mouth
    
    for face_img in face_imgs:        
        idx = [30, 8, 45, 36, 48, 54]
        face_2d = []
        face_3d = []
        for i in idx:
            x, y, z = face_img.landmark_3d_68[i]
            x = x * image.shape[1] / resized_img.shape[1]
            y = y * image.shape[0] / resized_img.shape[0]
            z = z * image.shape[1] / resized_img.shape[1]
            cv2.circle(image, (int(x), int(y)), 2, Color.red, 6)
            
            face_2d.append((int(x), int(y)))
            face_3d.append((int(x), int(y), int(z)))
        face_2d = np.array(face_2d, dtype='double')
        (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1200.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        p1 = (int(face_2d[0][0]), int(face_2d[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255,0,0), 2)
        
def main(model: DetectFasion, face_detect_model: FaceAnalysis, person_detect_model: YOLO):    
    cap1 = cv2.VideoCapture('eye.mp4')
    cap2 = cv2.VideoCapture(0)
    id_dict = {}
    frame_skip = 5  # ตั้งค่า frame skip
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        for _ in range(frame_skip - 1):
            cap1.grab()
            cap2.grab()

        detect_person_cam1(model, frame1, id_dict, 0)
        detect_face_cam2(person_detect_model, frame2, face_detect_model)

        if frame1.shape[0] != frame2.shape[0]:
            height, width, _ = frame1.shape
            frame1_ratio = width / height

            height, width, _ = frame2.shape
            frame2_ratio = width / height

            if frame1_ratio < frame2_ratio:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]), interpolation=cv2.INTER_AREA)

        frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
        frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
        merged_frame = cv2.vconcat([frame1, frame2])
        cv2.imshow('Test', merged_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    load_dotenv()
    global token, location
    token = ''
    location = 'KMUTT'
    # load_dotenv()
    # if os.getenv("ACCESS_TOKEN") is None:
    #     LoginGUI()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    personModel = YOLO(r"models\yolov8x.pt").to(device)
    fasionModel = YOLO(r"models\fashion_test.pt").to(device)
    sort_tracker = create_tracker('strongsort', get_tracker_config('strongsort'), Path('models\osnet_x0_25_msmt17.pt'), device, False, False)
    model = DetectFasion(personModel, fasionModel, sort_tracker)
    # face_detect_model = DetectFace()
    face_detect_model = FaceAnalysis(root='./', providers=['CPUExecutionProvider'])
    face_detect_model.prepare(ctx_id=0, det_size=(640, 640))
    # os.system('cls')
    main(model, face_detect_model, YOLO(r"models\yolov8n.pt").to(device))
