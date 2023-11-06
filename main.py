import cv2
from face_detect import face_detect
from face_mesh import FaceMesh
from person_detect import PersonDetect
from utils import FocusPoint
import time

cap = cv2.VideoCapture('test.mp4')
# cap = cv2.VideoCapture(0)
focus = FocusPoint()

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        focus.set_focus(x, y)

def main():
    try:
        cv2.setWindowTitle("MediaPipe FaceMesh", "MediaPipe FaceMesh")
        cv2.setMouseCallback("MediaPipe FaceMesh", draw_circle)
        while cap.isOpened():
            start = time.time()
            success, image = cap.read()
            # image = cv2.flip(image, 1)
            img_h, img_w, _ = image.shape
            if not success:
                print("Ignoring empty camera frame.")
                break
            person_imgs = PersonDetect(image)
            face_imgs, bounding_boxs = face_detect(image, person_imgs)
            if face_imgs and bounding_boxs:
                focus_x, focus_y = focus.get_focus()
                FaceMesh(image, face_imgs, person_imgs, bounding_boxs, focus_x, focus_y)
            # find fps
            end = time.time()
            fps = int(1 / (end - start))
            cv2.putText(image, f"{fps} FPS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("MediaPipe FaceMesh", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(e)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()