import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from src.predict import EmotionPredictor


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    predictor = EmotionPredictor(
        "models/deepfer_model.keras"
    )

    return predictor


predictor = load_model()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("😊 DeepFER")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📤 Predict Emotion",

        "📊 Model Performance",

        "ℹ About"

    ]

)


# =====================================================
# HOME
# =====================================================

if page == "🏠 Home":

    st.title("😊 DeepFER")

    st.subheader("Facial Emotion Recognition using Deep Learning")

    st.markdown("---")

    st.write("""

DeepFER is a Deep Learning based Facial Emotion Recognition system.

### Supported Emotions

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😀 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

### Technologies Used

- TensorFlow
- CNN
- OpenCV
- Streamlit
- NumPy
- Pillow

### Dataset

FER2013

### Model Accuracy

**61.49%**

""")

    st.image(
        "models/accuracy.png",
        caption="Training Accuracy"
    )



# =====================================================
# PREDICTION
# =====================================================

elif page == "📤 Predict Emotion":

    st.title("Upload Face Image")

    uploaded_file = st.file_uploader(

        "Choose an Image",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        output, results = predictor.predict_from_image(image)

        output = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                output,
                caption="Detected Face",
                width=350
            )

        if len(results) == 0:

            st.error("No Face Detected")

        else:

            st.success("Prediction Successful")

            for result in results:

                st.subheader(result["emotion"])

                st.progress(
                    float(result["confidence"])
                )

                st.write(

                    f"Confidence : {result['confidence']*100:.2f}%"

                )



# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "📊 Model Performance":

    st.title("CNN Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        if os.path.exists("models/accuracy.png"):

            st.image(
                "models/accuracy.png",
                caption="Accuracy"
            )

    with col2:

        if os.path.exists("models/loss.png"):

            st.image(
                "models/loss.png",
                caption="Loss"
            )

    st.markdown("---")

    st.metric(

        "Test Accuracy",

        "61.49%"

    )



# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ About":

    st.title("About DeepFER")

    st.write("""

### Facial Emotion Recognition

DeepFER is a CNN based Deep Learning model that detects human emotions from facial images.

### Developer

Aman Kumar

### Frameworks

- TensorFlow
- Keras
- OpenCV
- Streamlit

### Dataset

FER2013

### Internship Project

Paid Internship Project

""")