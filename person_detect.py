from ultralytics import YOLO
import cv2
import math

# load model
model = YOLO("models\yolov5su.pt").cpu()

def PersonDetect(image: cv2.Mat):
    # detect persons
    persons = model(image, classes=0)
    person_img = []
    # loop over persons
    for person in persons:
        # bounding boxes
        boxes = person.boxes
        # for each bounding box
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # crop person image
            person_img.append(image[y1:y2, x1:x2])
    # return person images
    return person_img
