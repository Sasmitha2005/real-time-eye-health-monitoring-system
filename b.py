# ...existing code...
import cv2
import numpy as np
import os

folder = r"C:\Users\sasth\OneDrive\Desktop\5th Semester\HDA\eye dataset"

# extensions to process
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for root, _, files in os.walk(folder):
    for file in files:
        if not file.lower().endswith(EXTS):
            continue

        path = os.path.join(root, file)
        image = cv2.imread(path)
        if image is None:
            print("Failed to load:", path)
            continue

        # optional resize for display speed
        max_width = 900
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # two HSV ranges for red (wrap-around)
        lower_red1 = np.array([0, 90, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 90, 70])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # compute redness percentage relative to non-black pixels (approximate visible area)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visible_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        visible_pixels = np.count_nonzero(visible_mask)
        red_pixels = np.count_nonzero(red_mask)
        redness_pct = (red_pixels / visible_pixels * 100) if visible_pixels > 0 else 0.0

        # overlay red mask on original
        overlay = image.copy()
        overlay[red_mask > 0] = (0, 0, 255)  # mark detected red areas in pure red (BGR)
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # show results
        cv2.imshow("Original", image)
        cv2.imshow("Redness Overlay", blended)
        cv2.imshow("Red Mask", red_mask)

        print(f"{path} -> redness: {redness_pct:.2f}% ({red_pixels} pixels)")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit

cv2.destroyAllWindows()
# ...existing code...