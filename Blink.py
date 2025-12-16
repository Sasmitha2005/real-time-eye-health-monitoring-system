"""
Invisible blink monitor.
Runs webcam in background (no preview window) and shows
a centered alert box if user doesn't blink for >10 seconds.
"""

import cv2, time, mediapipe as mp, numpy as np, pyautogui

NO_BLINK_SECONDS = 10.0
CLOSED_RATIO_THRESHOLD = 0.22

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# landmark indices
LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT = 159, 145, 33, 133
RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT = 386, 374, 263, 362

def norm_dist(a, b, w, h):
    return np.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def eye_ratio(lm, up, low, left, right, w, h):
    vert = norm_dist(lm[up], lm[low], w, h)
    horz = norm_dist(lm[left], lm[right], w, h)
    return vert / horz if horz else 1

def show_alert():
    pyautogui.alert("You can't blink more than 10 seconds!\nPlease blink 👀",
                    "Eye Blink Alert")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam"); return

    last_blink_time = time.time()
    eye_closed = False

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            l_ratio = eye_ratio(lm, LEFT_UPPER, LEFT_LOWER, LEFT_LEFT, LEFT_RIGHT, w, h)
            r_ratio = eye_ratio(lm, RIGHT_UPPER, RIGHT_LOWER, RIGHT_LEFT, RIGHT_RIGHT, w, h)
            l_open, r_open = l_ratio > CLOSED_RATIO_THRESHOLD, r_ratio > CLOSED_RATIO_THRESHOLD

            if not (l_open or r_open):  # both closed
                eye_closed = True
            else:
                if eye_closed:
                    last_blink_time = time.time()  # blink detected
                eye_closed = False

        if time.time() - last_blink_time > NO_BLINK_SECONDS:
            show_alert()
            last_blink_time = time.time()  # reset after alert

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit manually
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()