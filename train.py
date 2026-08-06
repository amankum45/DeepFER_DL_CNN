"""
train.py
-------------------------------------
Train CNN Model for Facial Emotion Recognition
Project: DeepFER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from preprocess import DataPreprocessor
from model import build_model


# =====================================================
# Configuration
# =====================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfer_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

data = DataPreprocessor(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

train_generator, validation_generator, test_generator = data.get_data_generators()

print("\nDataset Loaded Successfully!")

print("\nClass Labels")
print(train_generator.class_indices)

print(f"\nTraining Images   : {train_generator.samples}")
print(f"Validation Images : {validation_generator.samples}")
print(f"Testing Images    : {test_generator.samples}")


# =====================================================
# Build Model
# =====================================================

print("\n" + "=" * 60)
print("Building CNN Model...")
print("=" * 60)

model = build_model()

model.summary()


# =====================================================
# Callbacks
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# =====================================================
# Train Model
# =====================================================

print("\n" + "=" * 60)
print("Training Started...")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ],
    verbose=1
)


# =====================================================
# Save Training History
# =====================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(MODEL_DIR, "training_history.csv"),
    index=False
)


# =====================================================
# Evaluate Model
# =====================================================

print("\n" + "=" * 60)
print("Evaluating Model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")


# =====================================================
# Classification Report
# =====================================================

print("\nGenerating Classification Report...")

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

labels = list(test_generator.class_indices.keys())

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=labels
    )
)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")

print(cm)


# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.grid(True)

plt.savefig(os.path.join(MODEL_DIR, "accuracy.png"))

plt.show()


# =====================================================
# Loss Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.grid(True)

plt.savefig(os.path.join(MODEL_DIR, "loss.png"))

plt.show()


# =====================================================
# Completed
# =====================================================

print("\n" + "=" * 60)
print("Training Completed Successfully!")
print("=" * 60)

print(f"\nModel Saved           : {MODEL_PATH}")
print("Training History      : models/training_history.csv")
print("Accuracy Plot         : models/accuracy.png")
print("Loss Plot             : models/loss.png")