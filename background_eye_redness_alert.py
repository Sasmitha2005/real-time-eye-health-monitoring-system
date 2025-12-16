# background_eye_redness_alert.py
import cv2
import numpy as np
import tensorflow as tf
import time
import tkinter as tk
from tkinter import messagebox

# ✅ Load trained model
model = tf.keras.models.load_model("eye_redness_model.h5")

# ✅ Parameters
IMG_SIZE = (224, 224)
class_names = ["Normal", "Redness"]
ALERT_INTERVAL = 5 # seconds between alerts
CONFIDENCE_THRESHOLD = 0.50  # alert only if >50% sure

# ✅ Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

last_alert_time = 0
print("✅ Eye redness monitoring started (camera hidden). Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 🧩 Preprocess
        resized = cv2.resize(frame, IMG_SIZE)
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # 🧠 Predict
        predictions = model.predict(input_data, verbose=0)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        confidence = np.max(score)

        # 🚨 If redness detected with high confidence
        if class_index == 1 and confidence > CONFIDENCE_THRESHOLD:
            current_time = time.time()
            if current_time - last_alert_time > ALERT_INTERVAL:
                print(f"⚠️ Redness detected ({confidence*100:.2f}%) — showing alert!")
                last_alert_time = current_time

                # 🪟 Tkinter message box (like blink alert)
                root = tk.Tk()
                root.withdraw()  # hide the main window
                messagebox.showwarning(
                    "Eye Redness Alert",
                    f"Your eyes seem red ({confidence*100:.1f}% confidence)!\n"
                    "Please take a short break 👁️💧"
                )
                root.destroy()

        # Wait a short time (to reduce CPU load)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n👋 Monitoring stopped manually.")

cap.release()
cv2.destroyAllWindows()
