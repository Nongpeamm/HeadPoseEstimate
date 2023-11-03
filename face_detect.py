import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

def face_detect(image: cv2.Mat) -> [cv2.Mat]:
    tempImg = image.copy()
    tempImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    tempImg.flags.writeable = False

    results_face_detect = face_detection.process(tempImg)
    tempImg.flags.writeable = True

    tempImg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = tempImg.shape

    if results_face_detect.detections:
        face_imgs = []
        bounding_box_dicts = []
        for detection in results_face_detect.detections:
            # bounding box
            bounding_box_dict = {
                "xmin" : detection.location_data.relative_bounding_box.xmin,
                "ymin" : detection.location_data.relative_bounding_box.ymin,
                "width" : detection.location_data.relative_bounding_box.width,
                "height" : detection.location_data.relative_bounding_box.height
            }

            bounding_box_dicts.append(bounding_box_dict)
            if (bounding_box_dict["xmin"] < 0) or (bounding_box_dict['ymin'] < 0):
                continue

            cv2.rectangle(image, (int(bounding_box_dict["xmin"] * img_w), int(bounding_box_dict["ymin"] * img_h)),  # กว้าง  # สูง
                (int((bounding_box_dict["xmin"] + bounding_box_dict['width']) * img_w), int((bounding_box_dict['ymin'] + detection.location_data.relative_bounding_box.height)* img_h),),
                (255, 255, 255),2,
            )
            # get face image
            face_img = image[
                int(bounding_box_dict["ymin"] * img_h) : int((bounding_box_dict["ymin"] + bounding_box_dict['height']) * img_h),
                int(bounding_box_dict["xmin"] * img_w) : int((bounding_box_dict["xmin"] + bounding_box_dict["width"]) * img_w),
            ]
            face_imgs.append(face_img)
        return face_imgs, bounding_box_dicts
    else:
        cv2.putText(image, "No any face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return [], []