import os
import cv2
import requests
import torch
from threading import Thread
from boxmot import create_tracker, get_tracker_config
from dotenv import load_dotenv
from ultralytics import YOLO
from detect import DetectFasion, DetectFace, Tracker
from pathlib import Path
from colors import Color
from load_model import YoloDetectWithTracker
from insightface.app import FaceAnalysis
import numpy as np
import math

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

customer_dict = {}

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

def detect_face_cam2(person_detect_model: YOLO, image:cv2.Mat, face_detect_model: FaceAnalysis, track: Tracker):
    
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
    
    detections = []
    for face in (face_imgs):   
        look_pose = False
        face.bbox = [face.bbox[0] * image.shape[1] / resized_img.shape[1],
                         face.bbox[1] * image.shape[0] / resized_img.shape[0],
                         face.bbox[2] * image.shape[1] / resized_img.shape[1],
                         face.bbox[3] * image.shape[0] / resized_img.shape[0]]
        x1, y1, x2, y2 = face.bbox
        d_x1, d_y1, d_x2, d_y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (d_x1, d_y1), (d_x2, d_y2), Color.purple, 2)        
        pitch, yaw, roll = face.pose 
        if roll >= -10 and roll <= 10:
            look_pose = True
            cv2.putText(image, f"look detect", (d_x1, d_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        detection = [d_x1, d_y1, d_x2, d_y2, face.det_score, 0, look_pose]  
        detections.append(detection) 
        
    detections = np.array(detections)
    
    if detections.shape[0] > 0:
        dets = detections[:, 0:6]
        tracking = track.face_track(dets, image)
        
        for track in tracking:
            t_x1, t_y1, t_x2, t_y2 = track[0:4]
            id = track[4]
            if id not in customer_dict:
                customer_dict[id] = {"look_count": 0, "look_status": "not looking"}
            cv2.putText(image, f"{id}", (int(t_x1), int(t_y1 - 50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.rectangle(image, (int(t_x1), int(t_y1)), (int(t_x2), int(t_y2)), Color.green, 2)
            
            # calculate pose of face with detection bbox and track bbox
            for i, d_x1 in enumerate(detections[:, 0]):
                # print(f"t_x1: {int(t_x1)}, d_x1: {d_x1}")
                
                # check number distance between detection and track
                if abs(t_x1 - d_x1) < 10:
                    look_checker = detections[i][6]
                    if look_checker and look_checker == True and customer_dict[id]["look_status"] == "not looking":
                        customer_dict[id]["look_count"] = customer_dict[id]["look_count"] + 1
                        if customer_dict[id]["look_count"] > 5:
                            customer_dict[id]["look_status"] = "looking"
    print(customer_dict)        
    
        # idx = [30, 8, 45, 36, 48, 54]
        # face_2d = []
        # for id in idx:
        #     x, y, z = face.landmark_3d_68[id]
        #     x = x * image.shape[1] / resized_img.shape[1]
        #     y = y * image.shape[0] / resized_img.shape[0]
        #     if id == 45:
        #         x_text = int(x - 10)
        #         y_text = int(y - 10)
        #     # z = z * image.shape[1] / resized_img.shape[1]
        #     cv2.circle(image, (int(x), int(y)), 2, Color.red, 6)
            
        #     face_2d.append((int(x), int(y)))
        #     # face_3d.append((int(x), int(y), int(z)))
        # face_2d = np.array(face_2d, dtype='double')
        # (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        # p1 = (int(face_2d[0][0]), int(face_2d[0][1]))
        # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # cv2.line(image, p1, p2, (255,0,0), 2)
        
        # # check pose of face see the screen or not
        # rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        # proj_matrix = np.hstack((rvec_matrix, translation_vector))
        # eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        # pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        # pitch = math.degrees(math.asin(math.sin(pitch)))
        # yaw = math.degrees(math.asin(math.sin(yaw)))
        # roll = math.degrees(math.asin(math.sin(roll)))
            
        # if roll < -25:
        #     cv2.putText(image, f"-", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # elif roll > 25:
        #     cv2.putText(image, f"-", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(image, f"look detect", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


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
        detect_face_cam2(person_detect_model, frame2, face_detect_model, tracker_model)

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
    strong_sort_tracker_1 = create_tracker('strongsort', get_tracker_config('strongsort'), Path('models\osnet_x0_25_msmt17.pt'), device, False, False)
    strong_sort_tracker_2 = create_tracker('botsort', get_tracker_config('botsort'), Path("models\mobilenetv2_x1_0.pt"), device, False, False)
    model = DetectFasion(personModel, fasionModel, strong_sort_tracker_1)
    tracker_model = Tracker(strong_sort_tracker_2)
    # face_detect_model = DetectFace()
    face_detect_model = FaceAnalysis(root='./', providers=['CPUExecutionProvider'])
    face_detect_model.prepare(ctx_id=0, det_size=(640, 640))
    # os.system('cls')
    main(model, face_detect_model, YOLO(r"models\yolov8n.pt").to(device))
