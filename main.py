import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np

class VideoApp:
    def __init__(self, root, video_source, model_path, coco_path):
        self.root = root
        self.root.title("GZGKP Player")
        self.root.geometry("1100x580")

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100) #video ön belleğini arttırdım
        self.fps= self.cap.get(cv2.CAP_PROP_FPS)
        self.n_fps= self.fps*2
        self.cap.set(cv2.CAP_PROP_FPS,self.n_fps)

        self.canvas = tk.Canvas(self.root, width=1020, height=500)
        self.canvas.pack()

        self.btn_play = ttk.Button(self.root, text="Oynat", command=self.play)
        self.btn_play.place(x=100, y=515)

        self.btn_pause = ttk.Button(self.root, text="Pause", command=self.pause)
        self.btn_pause.place(x=200, y=515)

        self.btn_stop = ttk.Button(self.root, text="Stop", command=self.stop)
        self.btn_stop.place(x=300, y=515)

        self.giren_sayac = tk.Label(root, text="", fg="BLACK")
        self.giren_sayac.place(x=500, y=515)
        self.cikan_sayac = tk.Label(root, text="", fg="BLACK")
        self.cikan_sayac.place(x=600, y=515)

        self.giren_sayac["text"] = ("Giren", 0)
        self.cikan_sayac["text"] = ("Cikan", 0)

        self.is_playing = False
        self.model = YOLO(model_path)
        self.tracker = Tracker()
        self.g_count = []
        self.c_count = []
        self.g_box = [(610, 120), (610, 320), (1000, 130),(1000, 279)]
        self.c_box = [(610, 321),  (610, 520),(1000, 281), (1100, 520)]

        self.giren = {}
        self.cikan = {}
        self.class_list = self.load_class_list(coco_path)
        self.update()

    def load_class_list(self, coco_path):
        try:
            with open(coco_path, 'r') as coco_file:
                class_list = coco_file.read().split("\n")
            return class_list
        except FileNotFoundError:
            print(f"Error: COCO file not found at '{coco_path}'")
            return []

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        self.g_count = []
        self.c_count = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (1020, 500))
                results = self.model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                person_list = []
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])

                    coco_class = self.class_list[d]
                    if 'person' in coco_class:
                        person_list.append([x1, y1, x2, y2])
                update_id = self.tracker.update(person_list)
                for u_id in update_id:
                    xx1, yy1, xx2, yy2, id = u_id
                    result = cv2.pointPolygonTest(np.array(self.g_box, np.int32), (xx2, yy1), False)
                    if result >= 0:
                        self.giren[id] = (xx2, yy1)
                    if id in self.giren:
                        result1 = cv2.pointPolygonTest(np.array(self.c_box, np.int32), (xx2, yy1), False)
                        if result1 >= 0:
                            if self.g_count.count(id) == 0:
                                self.g_count.append(id)
                    result2 = cv2.pointPolygonTest(np.array(self.c_box, np.int32), (xx2, yy1), False)
                    if result2 >= 0:
                        self.cikan[id] = (xx2, yy1)
                    if id in self.cikan:
                        result3 = cv2.pointPolygonTest(np.array(self.g_box, np.int32), (xx2, yy1), False)
                        if result3 >= 0:
                            if self.c_count.count(id) == 0:
                                self.c_count.append(id)
                    cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'person', (xx1, yy1), 0.7, 1, colorR=0)
                    self.giren_sayac["text"]=("Giren",len(self.g_count))
                    self.cikan_sayac["text"] = ("Cikan",len(self.c_count))
                cv2.line(frame, (610, 320), (1000, 280), (0, 0, 255), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo

        self.root.after(10, self.update)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root, video_source="http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard", model_path='yolov8n.pt', coco_path='coco')
    root.mainloop()
