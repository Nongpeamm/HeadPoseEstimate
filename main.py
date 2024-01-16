import time
import torch
import cv2
import threading
from ultralytics import YOLO
from detect import DetectFasion, DetectFace
from load_model import YoloDetect
from colors import Color
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from pathlib import Path


def write_video(video_writer, image):
    video_writer.write(cv2.resize(image, (640, 480),
                       interpolation=cv2.INTER_AREA))


def main(model: DetectFasion):
    cap = cv2.VideoCapture('in.avi')
    video_writer = cv2.VideoWriter("test_beam.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    fashion_datas = dict()
    frame_skip = 5  # ตั้งค่า frame skip
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        start_time = time.time()
        # for _ in range(frame_skip - 1):
        #     cap.grab()

        person_imgs = model.person_detect(image)
        model.fashion_detect(person_imgs)
        # print(person_imgs)

        # face_imgs, bounding_boxs = face_detect(image, person_imgs)
        # if face_imgs and bounding_boxs:
        #     focus_x, focus_y = focus.get_focus()
        #     FaceMesh(image, face_imgs, person_imgs, bounding_boxs, focus_x, focus_y)
        stop_time = time.time()
        fps = 1 / (stop_time - start_time)
        cv2.putText(image, f"{fps:.1f} FPS", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, Color.green, 2)

        write_video(video_writer, image)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow("Fashion detect", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    video_writer.release()


def main2(person_detect_model: YOLO):
    face_detect_model = DetectFace()
    cap = cv2.VideoCapture(0)
    video_writer = cv2.VideoWriter(
        "test_beam2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        start_time = time.time()

        person_imgs, _ = YoloDetect(image, person_detect_model, classes=[0], conf=0.4, want_track_id=True)  # return person images

        for person_img in person_imgs:
            bbox_list, face_landmark = face_detect_model.detect(person_img)
            if bbox_list.shape[0] == 0 or face_landmark is None:
                continue

            bbox = bbox_list[0][0:4]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(person_img, (x1, y1), (x2, y2), Color.purple, 2)

            x, y = face_landmark[0][2] #nose
            # cv2.circle(person_img, (int(x), int(y)), 1, Color.red, 2) #error CUDA
            cv2.rectangle(person_img, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), Color.red, 2)

        stop_time = time.time()
        fps = 1 / (stop_time - start_time)
        cv2.putText(image, f"{fps:.1f} FPS", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, Color.green, 2)

        write_video(video_writer, image)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow("Face Detect", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    video_writer.release()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    personModel = YOLO(r"models\yolov8n.pt").to(device)
    fasionModel = YOLO(r"models\yolov8mOWNcctv80batch64AdamLR0.001.pt").to(device)
    sort_tracker = create_tracker('strongsort', get_tracker_config('strongsort'), Path('models\osnet_x0_25_msmt17.pt'), device, False, False)
    model = DetectFasion(personModel, fasionModel, sort_tracker)
    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=main, args=(model,))
    tracker_thread2 = threading.Thread(target=main2, args=(personModel,))
    # Start the tracker threads
    tracker_thread1.start()
    # tracker_thread2.start()

    # main(model)
    # main2(personModel)
    cv2.destroyAllWindows()
