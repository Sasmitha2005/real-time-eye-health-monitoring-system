# load_eye_dataset.py
import tensorflow as tf

# ✅ Path to your dataset
base_dir = r"C:\Users\sasth\OneDrive\Desktop\5th Semester\HDA\eye dataset\Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation\Uveitis"

train_dir = base_dir + "\\train"
val_dir = base_dir + "\\validation"

# ✅ Create training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),  # Resize all images
    batch_size=32,
    label_mode='categorical'
)

# ✅ Create validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# ✅ Normalize pixel values (0–255 → 0–1)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

print("✅ Dataset successfully loaded!")
print(f"Train batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")

# Optional: preview a few batches
for images, labels in train_ds.take(1):
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
