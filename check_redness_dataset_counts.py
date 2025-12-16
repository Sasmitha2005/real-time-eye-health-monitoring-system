# check_dataset_counts.py
import os

base_dir = r"C:\Users\sasth\OneDrive\Desktop\5th Semester\HDA\eye dataset\Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation\Uveitis"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

train_red = len(os.listdir(os.path.join(train_dir, "Redness")))
train_norm = len(os.listdir(os.path.join(train_dir, "Normal")))
val_red = len(os.listdir(os.path.join(val_dir, "Redness")))
val_norm = len(os.listdir(os.path.join(val_dir, "Normal")))

print(f"🧠 Train - Redness: {train_red}, Normal: {train_norm}")
print(f"🧪 Validation - Redness: {val_red}, Normal: {val_norm}")
print(f"📊 Total images: {train_red + train_norm + val_red + val_norm}")
