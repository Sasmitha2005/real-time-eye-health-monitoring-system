import os

base_dir = r"C:\Users\sasth\OneDrive\Desktop\5th Semester\HDA\eye dataset\Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation\DarkCircle"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

def count_images(folder):
    return len(os.listdir(folder)) if os.path.exists(folder) else 0

train_alert = count_images(os.path.join(train_dir, "alert"))
train_nonvig = count_images(os.path.join(train_dir, "non_vigilant"))
train_tired = count_images(os.path.join(train_dir, "tired"))

val_alert = count_images(os.path.join(val_dir, "alert"))
val_nonvig = count_images(os.path.join(val_dir, "non_vigilant"))
val_tired = count_images(os.path.join(val_dir, "tired"))

test_alert = count_images(os.path.join(test_dir, "alert"))
test_nonvig = count_images(os.path.join(test_dir, "non_vigilant"))
test_tired = count_images(os.path.join(test_dir, "tired"))

print("🧠 TRAIN COUNTS:")
print(f"   alert: {train_alert}, non_vigilant: {train_nonvig}, tired: {train_tired}")
print("🧪 VALIDATION COUNTS:")
print(f"   alert: {val_alert}, non_vigilant: {val_nonvig}, tired: {val_tired}")
print("🧩 TEST COUNTS:")
print(f"   alert: {test_alert}, non_vigilant: {test_nonvig}, tired: {test_tired}")

total = train_alert + train_nonvig + train_tired + val_alert + val_nonvig + val_tired + test_alert + test_nonvig + test_tired
print(f"📊 Total images: {total}")
