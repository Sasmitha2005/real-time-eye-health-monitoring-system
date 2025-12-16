import cv2
import numpy as np
import tensorflow as tf
import time
import tkinter as tk
from tkinter import messagebox

# ✅ Load trained model
model = tf.keras.models.load_model("eye_fatigue_model.h5")

# ✅ Parameters
IMG_SIZE = (224, 224)
class_names = ["alert", "non_vigilant", "tired"]
ALERT_INTERVAL = 7  # seconds
CONFIDENCE_THRESHOLD = 0.55

# ✅ Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

last_alert_time = 0
print("✅ Eye fatigue monitoring started. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        resized = cv2.resize(frame, IMG_SIZE)
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # 🧠 Predict
        predictions = model.predict(input_data, verbose=0)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        confidence = np.max(score)

        # 🚨 Alert logic
        if class_index == 2 and confidence > CONFIDENCE_THRESHOLD:  # 'tired' class
            current_time = time.time()
            if current_time - last_alert_time > ALERT_INTERVAL:
                print(f"⚠️ Fatigue detected ({confidence*100:.2f}%) — showing alert!")
                last_alert_time = current_time

                root = tk.Tk()
                root.withdraw()
                messagebox.showwarning(
                    "Eye Fatigue Alert",
                    f"You're showing signs of fatigue ({confidence*100:.1f}% confidence)!\n"
                    "Please blink, look away from the screen, and rest for a few seconds 👀💤"
                )
                root.destroy()

        time.sleep(1)

except KeyboardInterrupt:
    print("\n👋 Monitoring stopped manually.")
    cap.release()
    cv2.destroyAllWindows()
