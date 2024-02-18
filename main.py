import time
import torch
import cv2
import threading
import tkinter as tk
import os
import requests
import re
from dotenv import load_dotenv
from ultralytics import YOLO
from detect import DetectFasion, DetectFace
from load_model import YoloDetectWithTracker
from colors import Color
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from pathlib import Path


def write_video(video_writer, image):
    video_writer.write(cv2.resize(image, (640, 480),
                       interpolation=cv2.INTER_AREA))


def main(model: DetectFasion):
    cap = cv2.VideoCapture('piam.mp4')
    video_writer = cv2.VideoWriter(
        "test_beam.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    id_dict = {}
    DICT_MAX_SIZE = 3

    frame_skip = 5  # ตั้งค่า frame skip
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        start_time = time.time()
        for _ in range(frame_skip - 1):
            cap.grab()

        person_imgs, track_ids = model.person_detect(image)
        for track_id in track_ids:
            if track_id not in id_dict:
                # if len(id_dict) > DICT_MAX_SIZE:
                #     for data in id_dict.keys():
                #         all_top = dict(list(id_dict[data].items())[:5])
                #         max_top = max(all_top, key=all_top.get)
                #         max_pant = None
                #         if max_top != "dress":
                #             all_pant = dict(list(id_dict[data].items())[5:])
                #             max_pant = max(all_pant, key=all_pant.get)
                #         # print(f"ID: {data} Top: {max_top} : {id_dict[data][max_top]} Pant: {max_pant} : {id_dict[data][max_pant]}")
                #     requests.post(os.getenv("API_URL") + "/fashion", json={
                #         "top": max_top,
                #         "pant": max_pant,
                #         "location": location,
                #     }, headers={ "Authorization": token })
                #     id_dict.clear()

                id_dict[track_id] = {
                    "dress": 0,
                    "t-shirt": 0,
                    "jacket": 0,
                    "top": 0,
                    "long-sleeve": 0,
                    "short": 0,
                    "skirt": 0,
                    "trouser": 0
                }

        for person_img, track_id in zip(person_imgs, track_ids):
            _, fashion_labels = model.fashion_detect(person_img)
            for fashion_label in fashion_labels:
                id_dict[track_id][fashion_label] += 1

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


def main2(person_detect_model: YOLO, sort_tracker):
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

        person_imgs, _ = YoloDetectWithTracker(
            image, person_detect_model, sort_tracker, conf=0.4)  # return person images

        for person_img in person_imgs:
            bbox_list, face_landmark = face_detect_model.detect(person_img)
            if bbox_list is None or bbox_list.shape[0] == 0 or face_landmark is None:
                continue

            bbox = bbox_list[0][0:4]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(person_img, (x1, y1), (x2, y2), Color.purple, 2)

            x, y = face_landmark[0][2]  # nose
            # cv2.circle(person_img, (int(x), int(y)), 1, Color.red, 2) #error CUDA
            cv2.rectangle(person_img, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), Color.red, 2)

        stop_time = time.time()
        fps = 1 / (stop_time - start_time)
        cv2.putText(image, f"{fps:.1f} FPS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, Color.green, 2)

        write_video(video_writer, image)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow("Face Detect", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    video_writer.release()


def LoginGUI():
    def login():
        email = email_entry.get()
        password = password_entry.get()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            error_label.config(text="Email is not valid", bg="white")
            return

        url = os.getenv("API_URL") + "/auth/login"
        data = {
            "email": email,
            "password": password
        }
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                error_label.config(text="", bg="light blue")
                token = response.json()["accesstoken"]
                with open(".env", "a") as f:
                    f.write(f"ACCESS_TOKEN={token}\n")
                    root.destroy()
                    f.close()
            elif response.status_code == 400:
                error_label.config(text="Invalid email or password", bg="white")
            else:
                error_label.config(text="Something went wrong", bg="white")
        except requests.exceptions.RequestException as e:
            error_label.config(text="Connection error", bg="white")

    root = tk.Tk()
    root.title("Login")
    root.geometry("300x200")
    root.resizable(False, False)
    root.configure(bg="light blue")

    email_label = tk.Label(root, text="Email", bg="light blue")
    email_label.pack(pady=[10, 0])

    email_entry = tk.Entry(root, width=30)
    email_entry.pack()
    email_entry.focus()

    password_label = tk.Label(root, text="Password", bg="light blue")
    password_label.pack(pady=[10, 0])

    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack()

    login_button = tk.Button(root, text="Login", width=10, command=login)
    login_button.pack(pady=10)

    error_label = tk.Label(root, text="", fg="red", width=30, bg="light blue")
    error_label.pack(pady=5)
    root.mainloop()

    return


if __name__ == "__main__":
    # load_dotenv()
    # global token, location
    # token = ''
    # location = ''
    # if os.getenv("ACCESS_TOKEN") is None:
    #     LoginGUI()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    personModel = YOLO(r"models\yolov8n.pt").to(device)
    fasionModel = YOLO(r"models\fashion.pt").to(device)
    sort_tracker = create_tracker('strongsort', get_tracker_config('strongsort'), Path('models\osnet_x0_25_msmt17.pt'), device, False, False)
    model = DetectFasion(personModel, fasionModel, sort_tracker)
    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=main, args=(model,))
    tracker_thread2 = threading.Thread(target=main2, args=(personModel, sort_tracker,))
    # Start the tracker threads
    tracker_thread1.start()  # fashion detect
    # tracker_thread2.start() # face detect
    cv2.destroyAllWindows()
