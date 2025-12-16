import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import tkinter as tk
from tkinter import messagebox
import sqlite3
import winsound
import datetime

# =========================
# DATABASE (BACKEND)
# =========================
DB_NAME = "eye_health.db"

def get_db():
    return sqlite3.connect(DB_NAME)

def create_tables():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS User(
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS Alert(
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        alert_type TEXT,
        alert_time TEXT
    )
    """)

    conn.commit()
    conn.close()

# =========================
# ALERT FUNCTIONS
# =========================
def play_alert_sound():
    winsound.Beep(1000, 500)

def show_alert(title, message, user_id=None):
    play_alert_sound()

    if user_id:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Alert (user_id, alert_type, alert_time) VALUES (?, ?, ?)",
            (user_id, title, datetime.datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()

# =========================
# LOGIN & REGISTER (FRONTEND)
# =========================
class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Health Monitoring - Login")

        tk.Label(root, text="Username").grid(row=0, column=0)
        tk.Label(root, text="Password").grid(row=1, column=0)

        self.username = tk.Entry(root)
        self.password = tk.Entry(root, show="*")

        self.username.grid(row=0, column=1)
        self.password.grid(row=1, column=1)

        tk.Button(root, text="Login", command=self.login).grid(row=2, columnspan=2)
        tk.Button(root, text="Register", command=self.register).grid(row=3, columnspan=2)

    def register(self):
        user = self.username.get()
        pwd = self.password.get()

        if not user or not pwd:
            messagebox.showerror("Error", "All fields required")
            return

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("INSERT INTO User (username, password) VALUES (?, ?)", (user, pwd))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Registration Successful")
        except:
            messagebox.showerror("Error", "Username already exists")

    def login(self):
        user = self.username.get()
        pwd = self.password.get()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM User WHERE username=? AND password=?", (user, pwd))
        result = cur.fetchone()
        conn.close()

        if result:
            self.root.destroy()
            start_eye_monitoring(result[0])
        else:
            messagebox.showerror("Error", "Invalid credentials")

# =========================
# EYE MONITORING SYSTEM
# =========================
MIN_DISTANCE_CM = 10
NO_BLINK_SECONDS = 10
CLOSED_RATIO_THRESHOLD = 0.22

KNOWN_DISTANCE_CM = 30
KNOWN_FACE_WIDTH_PX = 120

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

FACE_LEFT = 234
FACE_RIGHT = 454
LEFT_UP, LEFT_DOWN, LEFT_L, LEFT_R = 159, 145, 33, 133
RIGHT_UP, RIGHT_DOWN, RIGHT_L, RIGHT_R = 386, 374, 263, 362

def euclidean(a, b, w, h):
    return np.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def estimate_distance(face_width):
    return (KNOWN_DISTANCE_CM * KNOWN_FACE_WIDTH_PX) / face_width

def eye_ratio(lm, up, down, left, right, w, h):
    v = euclidean(lm[up], lm[down], w, h)
    h1 = euclidean(lm[left], lm[right], w, h)
    return v / h1 if h1 else 1

def start_eye_monitoring(user_id):
    cap = cv2.VideoCapture(0)
    last_blink = time.time()
    eye_closed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            face_w = euclidean(lm[FACE_LEFT], lm[FACE_RIGHT], w, h)
            dist = estimate_distance(face_w)

            if dist < MIN_DISTANCE_CM:
                show_alert("Distance Alert", "Too close to screen", user_id)

            l = eye_ratio(lm, LEFT_UP, LEFT_DOWN, LEFT_L, LEFT_R, w, h)
            r = eye_ratio(lm, RIGHT_UP, RIGHT_DOWN, RIGHT_L, RIGHT_R, w, h)

            if l < CLOSED_RATIO_THRESHOLD and r < CLOSED_RATIO_THRESHOLD:
                eye_closed = True
            else:
                if eye_closed:
                    last_blink = time.time()
                eye_closed = False

        if time.time() - last_blink > NO_BLINK_SECONDS:
            show_alert("Blink Alert", "No blink detected", user_id)
            last_blink = time.time()

        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    create_tables()
    root = tk.Tk()
    LoginApp(root)
    root.mainloop()
