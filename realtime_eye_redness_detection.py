# realtime_eye_redness_detection.py
import cv2
import numpy as np
import tensorflow as tf

# ✅ Load your trained model
model = tf.keras.models.load_model("eye_redness_model.h5")

# ✅ Define image size used during training
IMG_SIZE = (224, 224)

# ✅ Define class names in the same order as training
class_names = ["Normal", "Redness"]

# ✅ Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access the webcam.")
    exit()

print("✅ Real-time eye redness detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    # 🟦 Optional: Flip horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # 🧩 Preprocess the frame
    resized = cv2.resize(frame, IMG_SIZE)
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    # 🧠 Predict
    predictions = model.predict(input_data)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)

    # 🟥 Display result
    label = f"{class_names[class_index]} ({confidence:.2f}%)"
    color = (0, 255, 0) if class_index == 0 else (0, 0, 255)

    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    cv2.imshow("Eye Redness Detection", frame)

    # 🛑 Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
