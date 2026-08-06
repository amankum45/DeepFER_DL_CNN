import os
import cv2
from src.predict import EmotionPredictor

MODEL_PATH = "models/deepfer_model.keras"
IMAGE_PATH = "images/test.jpg"

print("Current Directory :", os.getcwd())
print("Model Exists      :", os.path.exists(MODEL_PATH))
print("Image Exists      :", os.path.exists(IMAGE_PATH))

image = cv2.imread(IMAGE_PATH)

print("Image Loaded      :", image is not None)

predictor = EmotionPredictor(MODEL_PATH)

results = predictor.predict(image)

if len(results) == 0:
    print("❌ No face detected.")
else:
    for result in results:

        x, y, w, h = result["box"]

        emotion = result["emotion"]
        confidence = result["confidence"]

        print(f"Emotion : {emotion}")
        print(f"Confidence : {confidence*100:.2f}%")

        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            image,
            f"{emotion} ({confidence*100:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

cv2.imshow("DeepFER", image)
cv2.waitKey(0)
cv2.destroyAllWindows()