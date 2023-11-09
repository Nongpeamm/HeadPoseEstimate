import cv2

def YoloDetect(image: cv2.Mat, model, classes=None):
    # detect persons
    objs = model(image, classes=classes)
    obj_img = []
    # loop over persons
    for obj in objs:
        # bounding boxes
        boxes = obj.boxes
        # for each bounding box
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # draw label
            cv2.putText(image, f"{obj.names[0]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # crop obj image
            obj_img.append(image[y1:y2, x1:x2])
    # return obj images
    return obj_img
