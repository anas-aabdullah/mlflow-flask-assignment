import os
import mlflow
import mlflow.keras
from mlflow import log_metric, log_param, log_artifact
import dagshub
import tensorflow as tf
from tensorflow.keras import layers, models # pyright: ignore[reportMissingImports]

# =====================================================
# 1) CONNECT TO DAGSHUB MLflow SERVER
# =====================================================
dagshub.init(
    repo_owner="anas-aabdullah",
    repo_name="mlflow-flask-assignment",
    mlflow=True
)

# =====================================================
# 2) DATASET LOCATION
# =====================================================
DATASET_PATH = "flower_photos"

if not os.path.exists(DATASET_PATH):
    print("Downloading dataset...")
    tf.keras.utils.get_file(
        "flower_photos.tgz",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        extract=True
    )
else:
    print("Dataset already exists, skipping download.")

# Count images
for cls in os.listdir(DATASET_PATH):
    p = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(p):
        print(f"{cls}: {len(os.listdir(p))} images")

# =====================================================
# 3) DATA PIPELINE
# =====================================================
img_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# =====================================================
# 4) MODEL
# =====================================================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# 5) MLflow Run
# =====================================================
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("image_size", img_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_classes", len(class_names))
    mlflow.log_param("model_type", "Simple CNN")

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Log metrics
    mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])

    # =================================================
    # 🔥 SAVE MODEL LOCALLY
    # =================================================
    model.save("model.h5")
    print("Model saved as model.h5")

    # =================================================
    # 🔥 UPLOAD model.h5 TO DAGSHUB AS ARTIFACT
    # =================================================
    mlflow.log_artifact("model.h5")

print("\nTraining complete. Model uploaded to DagsHub artifacts.")
