import time, torch, cv2
from load_model import YoloDetect, OnnxDetect
from focuspoint import FocusPoint
from ultralytics import YOLO

cap = cv2.VideoCapture('test.mp4')
# cap = cv2.VideoCapture(0)
focus = FocusPoint()
# personModel = torch.hub.load('ultralytics/yolov5', 'custom', 'models\yolov5s_openvino_model', device='cpu')
personModel = YOLO(r"models\yolov8n.pt").cuda()
# fasionModel = YOLO(r"models\yolov8sfashion40batch32.pt").cuda()
video_writer = cv2.VideoWriter("test_beam.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        focus.set_focus(x, y)

def main():
    from face_detect import face_detect
    from face_mesh import FaceMesh
    try:
        cv2.setWindowTitle("MediaPipe FaceMesh", "MediaPipe FaceMesh")
        # cv2.setMouseCallback("MediaPipe FaceMesh", draw_circle)
        while cap.isOpened():
            start = time.time()
            success, image = cap.read()
            #filp image
            image = cv2.flip(image, 1)
            img_h, img_w, _ = image.shape
            focus.set_focus(int(img_w / 2), int(img_h / 2))
            if not success:
                print("Ignoring empty camera frame.")
                break
            # person_imgs = OnnxDetect(image, personModel)
            person_imgs = YoloDetect(image, personModel, classes=[0], conf=0.4)
            # for person_img in person_imgs:
            #     YoloDetect(person_img, fasionModel, verbose=True)
            face_imgs, bounding_boxs = face_detect(image, person_imgs)
            if face_imgs and bounding_boxs:
                focus_x, focus_y = focus.get_focus()
                FaceMesh(image, face_imgs, person_imgs, bounding_boxs, focus_x, focus_y)
            # find fps
            end = time.time()
            fps = int(1 / (end - start))
            cv2.putText(image, f"{fps} FPS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            resized_image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

            # write video
            video_writer.write(cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA))

            cv2.imshow("MediaPipe FaceMesh", resized_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(e)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()