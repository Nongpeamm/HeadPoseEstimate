import cv2
from colors import Color
import numpy as np


def YoloDetect(image: cv2.Mat, model, classes=None, conf=0.25, verbose=False, color: tuple = Color.white):
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return []
    # detect persons
    objs = model.track(image, classes=classes, conf=conf,
                       verbose=verbose, persist=True, tracker='bytetrack.yaml')
    obj_img = []
    # bounding boxes
    boxes = objs[0].boxes
    if boxes.id is None:
        return obj_img
    classes = boxes.cls.int().tolist()
    # for each bounding box
    for box, labels in zip(boxes, classes):
        # get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # draw bounding box over the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # draw label
        cv2.putText(image, objs[0].names[labels], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # get person image
        obj_img.append(image[y1:y2, x1:x2])

    return obj_img

def YoloDetectWithTracker(image: cv2.Mat, model, sort_tracker, conf=0.25):
    # Detect persons using YOLO model
    objs = model.track(image, classes=0, conf=conf,
                       verbose=False, persist=True, tracker='bytetrack.yaml')

    obj_img = []

    # Check if there are any detections
    if objs[0].boxes.id is None:
        return obj_img

    # Extract information from YOLO detections
    boxes = objs[0].boxes
    xyxys = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int)
    detections = []

    for xyxy, conf, cls, track_id in zip(xyxys, confs, classes, ids):
        x1, y1, x2, y2 = xyxy

        # Format detections as expected by the SORT tracker
        detections.append([x1, y1, x2, y2, conf, cls])

    # Update the SORT tracker
    tracks = sort_tracker.update(np.array(detections), image)
    for track in tracks:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = track[0:4].astype(int)
        track_id = int(track[4])
        # Draw the bounding box over the image
        cv2.rectangle(image, (x1, y1), (x2, y2), Color.white, 2)
        # Draw the label
        cv2.putText(image, objs[0].names[track[6]], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, Color.white, 2)
        # Draw the tracking id
        cv2.putText(image, f'id: {track_id}', (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, Color.white, 2)
        # Get the person image
        obj_img.append(image[y1:y2, x1:x2])

    return obj_img

def OnnxDetect(image: cv2.Mat, model):
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    # detect persons
    objs = model(image)
    for obj in objs.pandas().xyxy:
        persons_obj = obj[obj["class"] == 0]
        print(persons_obj)
