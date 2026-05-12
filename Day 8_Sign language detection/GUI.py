import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image, ImageTk
from collections import deque, Counter

# =====================
# Paths
# =====================
MODEL_PATH = r"E:\Computer Vision in 30 Days\Computer-Vision-in-30-Days\Day 8_Sign language detection\cnn_model_final.h5"
LABEL_PATH = r"E:\Computer Vision in 30 Days\Computer-Vision-in-30-Days\Day 8_Sign language detection\label_map.pkl"

# =====================
# Load Model
# =====================
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found")
    exit()

model = load_model(MODEL_PATH)
label_map = joblib.load(LABEL_PATH)
inv_map = {v: k for k, v in label_map.items()}

print("✅ Model Loaded Successfully")

# =====================
# MediaPipe
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =====================
# App
# =====================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detector")
        self.root.geometry("950x650")
        self.root.configure(bg="#1e1e1e")

        self.cap = None
        self.running = False
        self.pred_history = deque(maxlen=10)

        Label(root, text="Sign Language Recognition",
              font=("Arial", 22, "bold"),
              fg="white", bg="#1e1e1e").pack(pady=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.pred_label = Label(root, text="Prediction: ---",
                                font=("Arial", 18),
                                fg="#00ffcc", bg="#1e1e1e")
        self.pred_label.pack(pady=10)

        btn_frame = Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        Button(btn_frame, text="Start",
               command=self.start_camera,
               bg="#28a745", fg="white",
               width=12).grid(row=0, column=0, padx=10)

        Button(btn_frame, text="Stop",
               command=self.stop_camera,
               bg="#dc3545", fg="white",
               width=12).grid(row=0, column=1, padx=10)

    def preprocess(self, img):
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                h, w, _ = frame.shape
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                xmin = max(0, min(x_list) - 30)
                ymin = max(0, min(y_list) - 30)
                xmax = min(w, max(x_list) + 30)
                ymax = min(h, max(y_list) + 30)

                hand_img = frame[ymin:ymax, xmin:xmax]

                if hand_img.size != 0:
                    processed = self.preprocess(hand_img)

                    probs = model.predict(processed, verbose=0)[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred]

                    if confidence > 0.7:
                        real_label = inv_map.get(pred, pred)
                        letter = chr(real_label + 65)

                        self.pred_history.append(letter)
                        final_pred = Counter(self.pred_history).most_common(1)[0][0]

                        self.pred_label.config(
                            text=f"Prediction: {final_pred} ({confidence:.2f})"
                        )

                        cv2.putText(frame, final_pred,
                                    (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,255), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((750, 450))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

# =====================
# Run
# =====================
root = tk.Tk()
app = App(root)
root.mainloop()