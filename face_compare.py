import sys 
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import onnx
import onnxruntime as ort
import cv2
import mediapipe as mp

from face_class import Face_class

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

Face_class = Face_class()
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    frame_ori = frame.copy()
    
    mp_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_face.flags.writeable = False
    
    results_face_detect = face_detection.process(mp_face)
    mp_face.flags.writeable = True
    
    mp_face = cv2.cvtColor(mp_face, cv2.COLOR_RGB2BGR)
    
    if results_face_detect.detections:
        for detection in results_face_detect.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, width, height = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1, y1, width, height = int(x1 * 640), int(y1 * 480), int(width * 640), int(height * 480)
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    output_list = Face_class.detect(frame)
    
    width , height, channel = frame.shape
    # print(f'list of face detect {output_list} length : {len(output_list)}')
    bbox_list = output_list[0]
    landmark = output_list[1]
    
    for bbox_info in bbox_list:
        bbox = bbox_info[0:4]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        detect_draw = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print(f'bbox : {bbox}')
        
    for landmark_info in landmark:
        print(landmark_info) 
        for x, y in landmark_info:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
            
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()


