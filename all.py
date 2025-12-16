"""
=============================================================
  REAL-TIME EYE HEALTH MONITORING SYSTEM – ALL MODULES IN ONE
=============================================================

Modules included:
 1. Distance Monitoring (MediaPipe)
 2. Blink Monitoring (MediaPipe)
 3. Eye Fatigue Detection (TensorFlow Model)
 4. Eye Redness Detection (TensorFlow Model)
 
 Alerts shown through Tkinter MessageBox.
 Camera feed runs fully hidden (no preview window).
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import tkinter as tk
from tkinter import messagebox
import pyautogui

# -----------------------------------------
#          SETTINGS
# -----------------------------------------
MIN_DISTANCE_CM = 10
NO_BLINK_SECONDS = 10.0
CLOSED_RATIO_THRESHOLD = 0.22

DISTANCE_ALERT = "⚠️ You are too close to the screen! Please move back 👀"
BLINK_ALERT = "⚠️ You haven't blinked for 10 seconds! Please blink 👀"

KNOWN_DISTANCE_CM = 30
KNOWN_FACE_WIDTH_PX = 120

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD_FATIGUE = 0.55
CONFIDENCE_THRESHOLD_REDNESS = 0.80   # ✅ UPDATED TO 80%

FATIGUE_INTERVAL = 7
REDNESS_INTERVAL = 5

# -----------------------------------------
#          LOAD MODELS
# -----------------------------------------
fatigue_model = tf.keras.models.load_model("eye_fatigue_model.h5")
redness_model = tf.keras.models.load_model("eye_redness_model.h5")

# -----------------------------------------
#          MEDIAPIPE FACE MESH
# -----------------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Facial landmarks
FACE_LEFT = 234
FACE_RIGHT = 454
LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT = 159, 145, 33, 133
RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT = 386, 374, 263, 362


# -----------------------------------------
#          FUNCTIONS
# -----------------------------------------
def euclidean(a, b, w, h):
    return np.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def estimate_distance(face_width_px):
    return (KNOWN_DISTANCE_CM * KNOWN_FACE_WIDTH_PX) / face_width_px


def eye_ratio(lm, up, low, left, right, w, h):
    vert = euclidean(lm[up], lm[low], w, h)
    horz = euclidean(lm[left], lm[right], w, h)
    return vert / horz if horz else 1


def show_alert(title, message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()


# -----------------------------------------
#          MAIN FUNCTION
# -----------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    last_blink_time = time.time()
    eye_closed = False

    last_fatigue_alert = 0
    last_redness_alert = 0

    print("✅ Real-Time Eye Monitoring Started (Camera Hidden)")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- PROCESS FOR DISTANCE + BLINK ----
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ---- DISTANCE CHECK ----
            face_width_px = euclidean(lm[FACE_LEFT], lm[FACE_RIGHT], w, h)
            est_distance = estimate_distance(face_width_px)

            if est_distance < MIN_DISTANCE_CM:
                show_alert("Distance Alert", DISTANCE_ALERT)

            # ---- BLINK CHECK ----
            l_ratio = eye_ratio(lm, LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT, w, h)
            r_ratio = eye_ratio(lm, RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT, w, h)

            l_open = l_ratio > CLOSED_RATIO_THRESHOLD
            r_open = r_ratio > CLOSED_RATIO_THRESHOLD

            if not (l_open or r_open):
                eye_closed = True
            else:
                if eye_closed:
                    last_blink_time = time.time()
                eye_closed = False

        # If blink timeout exceeded → alert
        if time.time() - last_blink_time > NO_BLINK_SECONDS:
            show_alert("Blink Alert", BLINK_ALERT)
            last_blink_time = time.time()

        # ----------------------------------------------------
        #     FATIGUE + REDNESS MODEL PREDICTIONS
        # ----------------------------------------------------
        resized = cv2.resize(frame, IMG_SIZE)
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # -------- FATIGUE --------
        fat_pred = fatigue_model.predict(input_data, verbose=0)
        fat_score = tf.nn.softmax(fat_pred[0])
        fat_class = np.argmax(fat_score)
        fat_conf = np.max(fat_score)

        if fat_class == 2 and fat_conf > CONFIDENCE_THRESHOLD_FATIGUE:
            if time.time() - last_fatigue_alert > FATIGUE_INTERVAL:
                show_alert(
                    "Fatigue Alert",
                    f"You look tired ({fat_conf*100:.1f}% confidence). Please rest 😴"
                )
                last_fatigue_alert = time.time()

        # -------- REDNESS --------
        red_pred = redness_model.predict(input_data, verbose=0)
        red_score = tf.nn.softmax(red_pred[0])
        red_class = np.argmax(red_score)
        red_conf = np.max(red_score)

        # REDNESS ONLY IF > 80%
        if red_class == 1 and red_conf > CONFIDENCE_THRESHOLD_REDNESS:
            if time.time() - last_redness_alert > REDNESS_INTERVAL:
                show_alert(
                    "Redness Alert",
                    f"High redness detected ({red_conf*100:.1f}% confidence)!\n"
                    "Please take a break 👁️💧"
                )
                last_redness_alert = time.time()

        # Reduce CPU load
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------
#         RUN MAIN
# -----------------------------------------
if __name__ == "__main__":
    main()
