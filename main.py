import cv2
from face_detect import face_detect
from face_mesh import FaceMesh
from utils import FocusPoint
cap = cv2.VideoCapture(0)
focus = FocusPoint()

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        focus.set_focus(x, y)

def main():
    try:
        cv2.setWindowTitle('MediaPipe FaceMesh', 'MediaPipe FaceMesh')
        cv2.setMouseCallback('MediaPipe FaceMesh', draw_circle)
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                break
            face_imgs, bounding_boxs = face_detect(image)
            if face_imgs and bounding_boxs:
                img_h, img_w, _ = image.shape
                focus_x, focus_y = focus.get_focus()
                FaceMesh(image, face_imgs, bounding_boxs, img_w, img_h, focus_x, focus_y)
            cv2.imshow('MediaPipe FaceMesh', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()