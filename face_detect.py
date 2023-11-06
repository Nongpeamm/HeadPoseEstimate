import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

def face_detect(image: cv2.Mat, person_imgs: [cv2.Mat]) -> [cv2.Mat]:
    face_imgs = []
    bounding_box_dicts = []
    # each person image
    for person_img in person_imgs:
        # convert to RGB
        tempImg = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        tempImg.flags.writeable = False

        # face detection
        results_face_detect = face_detection.process(tempImg)
        tempImg.flags.writeable = True

        # convert back to BGR
        tempImg = cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = tempImg.shape

        # if face detected
        if results_face_detect.detections:
            # loop for each face
            for detection in results_face_detect.detections:
                # bounding box
                bounding_box_dict = {
                    "xmin" : detection.location_data.relative_bounding_box.xmin,
                    "ymin" : detection.location_data.relative_bounding_box.ymin,
                    "width" : detection.location_data.relative_bounding_box.width,
                    "height" : detection.location_data.relative_bounding_box.height
                }

                bounding_box_dicts.append(bounding_box_dict)
                # if bounding box is out of image
                if (bounding_box_dict["xmin"] < 0) or (bounding_box_dict['ymin'] < 0):
                    continue

                # draw face bounding box
                cv2.rectangle(person_img, (int(bounding_box_dict["xmin"] * img_w), int(bounding_box_dict["ymin"] * img_h)),  # กว้าง  # สูง
                    (int((bounding_box_dict["xmin"] + bounding_box_dict['width']) * img_w), int((bounding_box_dict['ymin'] + detection.location_data.relative_bounding_box.height)* img_h),),
                    (255, 255, 255),2,
                )
                # get face image
                face_img = person_img[
                    int(bounding_box_dict["ymin"] * img_h) : int((bounding_box_dict["ymin"] + bounding_box_dict['height']) * img_h),
                    int(bounding_box_dict["xmin"] * img_w) : int((bounding_box_dict["xmin"] + bounding_box_dict["width"]) * img_w),
                ]
                face_imgs.append(face_img)
                face_h, face_w, _ = face_img.shape
                
                # put face size text
                cv2.putText(person_img, f"{face_w}x{face_h}", (face_w, face_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No any face detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return face_imgs, bounding_box_dicts