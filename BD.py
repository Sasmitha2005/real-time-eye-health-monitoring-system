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
MIN_DISTANCE_CM = 10          
NO_BLINK_SECONDS = 10.0       
CLOSED_RATIO_THRESHOLD = 0.22 

DISTANCE_ALERT = "⚠️ You are too close to the screen! Please move back 👀"
BLINK_ALERT = "⚠️ You haven't blinked for 10 seconds! Please blink 👀"

# Calibration constants
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

# Landmark indices
FACE_LEFT, FACE_RIGHT = 234, 454
LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT = 159, 145, 33, 133
RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT = 386, 374, 263, 362


def euclidean(a, b, w, h):
    return np.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def estimate_distance(face_width_px):
    if face_width_px == 0:
        return 1000  # Avoid division error
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

            # Distance Monitoring
            face_width_px = euclidean(lm[FACE_LEFT], lm[FACE_RIGHT], w, h)
            dist = estimate_distance(face_width_px)

            if dist < MIN_DISTANCE_CM:
                show_alert(DISTANCE_ALERT)

            # Blink Monitoring
            left_ratio = eye_ratio(lm, LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT, w, h)
            right_ratio = eye_ratio(lm, RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT, w, h)

            eyes_open = left_ratio > CLOSED_RATIO_THRESHOLD or right_ratio > CLOSED_RATIO_THRESHOLD

            if not eyes_open:
                eye_closed = True
            else:
                if eye_closed:
                    last_blink_time = time.time()  # Blink detected
                eye_closed = False

        # Blink alert timer
        if time.time() - last_blink_time > NO_BLINK_SECONDS:
            show_alert(BLINK_ALERT)
            last_blink_time = time.time()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
