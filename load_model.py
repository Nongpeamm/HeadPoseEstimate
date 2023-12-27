import cv2
from colors import Color


def YoloDetect(image: cv2.Mat, model, classes=None, conf=0.25, verbose=False, color: tuple = Color.white, want_track_id=False):
    # detect persons
    objs = model.track(image, classes=classes, conf=conf,
                       verbose=verbose, persist=True, tracker='bytetrack.yaml')
    obj_img = []
    # bounding boxes
    boxes = objs[0].boxes
    if boxes.id is None:
        return obj_img
    tracking_ids = boxes.id.int().tolist()
    classes = boxes.cls.int().tolist()
    # for each bounding box
    for box, labels, tracking_id in zip(boxes, classes, tracking_ids):
        # get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # draw bounding box over the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # draw label
        cv2.putText(image, objs[0].names[labels], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if want_track_id:
            # draw tracking id
            cv2.putText(image, f'id: {tracking_id}', (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # get person image
        obj_img.append(image[y1:y2, x1:x2])

    return obj_img


def OnnxDetect(image: cv2.Mat, model):
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    # detect persons
    objs = model(image)
    for obj in objs.pandas().xyxy:
        persons_obj = obj[obj["class"] == 0]
        print(persons_obj)
