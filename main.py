import collections
import time, torch, cv2, numpy as np
from ultralytics import YOLO
from detect import DetectFasion

def write_video(video_writer, image):
    video_writer.write(cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA))

def main():
    processing_times = collections.deque()
    cap = cv2.VideoCapture('test.mp4')
    video_writer = cv2.VideoWriter("test_beam.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    personModel = YOLO(r"models\yolov8n.pt").to(device)
    fasionModel = YOLO(r"models\yolov8mfashion200.pt").to(device)
    model = DetectFasion(personModel, fasionModel)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        start_time = time.time()

        person_imgs = model.person_detect(image)
        model.fashion_detect(person_imgs)

        # face_imgs, bounding_boxs = face_detect(image, person_imgs)
        # if face_imgs and bounding_boxs:
        #     focus_x, focus_y = focus.get_focus()
        #     FaceMesh(image, face_imgs, person_imgs, bounding_boxs, focus_x, focus_y)


        stop_time = time.time()
        processing_times.append(stop_time - start_time)
        # Use processing times from last 200 frames.
        if len(processing_times) > 200:
            processing_times.popleft()
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(image, f"{fps:.1f} FPS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        write_video(video_writer, image)
        image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)
        cv2.imshow("MediaPipe FaceMesh", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()

if __name__ == "__main__":
    main()