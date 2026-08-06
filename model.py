"""
model.py
---------------------------------------
Improved CNN Model for DeepFER
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(48, 48, 1), num_classes=7):

    model = Sequential([

        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.30),

        # Block 4
        Conv2D(256, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.40),

        # Instead of Flatten()
        GlobalAveragePooling2D(),

        Dense(256, activation="relu"),
        Dropout(0.50),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model