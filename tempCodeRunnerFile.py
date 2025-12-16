"""
Real-Time Eye Health Monitoring System
1. Alerts if the user sits too close to the screen (< 10 cm).
2. Alerts if the user doesn't blink for more than 10 seconds.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# --- SETTINGS ---
MIN_DISTANCE_CM = 10          # Distance threshold for alert
NO_BLINK_SECONDS = 10.0       # Maximum allowed time without blinking
CLOSED_RATIO_THRESHOLD = 0.22 # Eye closure threshold

DISTANCE_ALERT = "⚠️ You are too close to the screen! Please move back 👀"
BLINK_ALERT = "⚠️ You haven't blinked for 10 seconds! Please blink 👀"

# Calibration constants for face width
KNOWN_DISTANCE_CM = 30
KNOWN_FACE_WIDTH_PX = 120

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
FACE_LEFT = 234
FACE_RIGHT = 454
LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT = 159, 145, 33, 133
RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT = 386, 374, 263, 362

def euclidean(a, b, w, h):
    return np.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def estimate_distance(face_width_px):
    return (KNOWN_DISTANCE_CM * KNOWN_FACE_WIDTH_PX) / face_width_px

def eye_ratio(lm, up, low, left, right, w, h):
    vert = euclidean(lm[up], lm[low], w, h)
    horz = euclidean(lm[left], lm[right], w, h)
    return vert / horz if horz else 1

def show_alert(message):
    pyautogui.alert(message, "Eye Health Alert")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    last_blink_time = time.time()
    eye_closed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ----- Distance Monitoring -----
            face_width_px = euclidean(lm[FACE_LEFT], lm[FACE_RIGHT], w, h)
            est_distance = estimate_distance(face_width_px)
            if est_distance < MIN_DISTANCE_CM:
                show_alert(DISTANCE_ALERT)

            # ----- Blink Monitoring -----
            l_ratio = eye_ratio(lm, LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT, w, h)
            r_ratio = eye_ratio(lm, RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT, w, h)
            l_open, r_open = l_ratio > CLOSED_RATIO_THRESHOLD, r_ratio > CLOSED_RATIO_THRESHOLD

            if not (l_open or r_open):
                eye_closed = True
            else:
                if eye_closed:
                    last_blink_time = time.time()  # Blink detected
                eye_closed = False

        # Check blink timer
        if time.time() - last_blink_time > NO_BLINK_SECONDS:
            show_alert(BLINK_ALERT)
            last_blink_time = time.time()  # Reset after alert

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
