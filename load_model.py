import cv2, torch, numpy as np

def YoloDetect(image: cv2.Mat, model, classes=None, conf=0.25, verbose=False):
    # detect persons
    objs = model.predict(image, classes=classes, conf=conf, verbose=verbose)
    obj_img = []
    # loop over persons
    for obj in objs:
        # bounding boxes
        boxes = obj.boxes
        # for each bounding box
        for box, labels in zip(boxes, boxes.cls):
            # get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # draw bounding box over the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw label
            cv2.putText(image, obj.names[int(labels.item())], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # get person image
            obj_img.append(image[y1:y2, x1:x2])
        
    return obj_img

def OnnxDetect(image:cv2.Mat, model):
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    # detect persons
    objs = model(image)
    for obj in objs.pandas().xyxy:
        persons_obj = obj[obj["class"] == 0]
        print(persons_obj)
