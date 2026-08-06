"""
predict.py
-------------------------------------
Emotion Prediction Module
Project: DeepFER
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model


class EmotionPredictor:

    def __init__(self, model_path):

        self.model = load_model(model_path)

        self.labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise"
        ]

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def predict_from_image(self, image):
        """
        Predict emotion from a BGR image.
        Returns:
            Annotated image and prediction results.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        results = []

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face.astype("float32") / 255.0

            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            prediction = self.model.predict(face, verbose=0)

            idx = np.argmax(prediction)

            confidence = float(prediction[0][idx])

            emotion = self.labels[idx]

            results.append({
                "emotion": emotion,
                "confidence": confidence,
                "box": (x, y, w, h)
            })

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"{emotion} ({confidence*100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return image, results