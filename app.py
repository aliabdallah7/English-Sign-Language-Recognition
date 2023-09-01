import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from io import BytesIO
from playsound import playsound
from gtts import gTTS

image = Image.open('./img/sign-language.png')
st.set_page_config(page_title='ESLR', page_icon=image)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Constants
MIN_DETECTION_CONFIDENCE = 0.3
MAX_SEQUENCE_LENGTH = 100
NUM_EXPECTED_FEATURES = 100

# Streamlit UI
st.title("Sign Language Detector")

# Upload a photo
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process the image with MediaPipe Hands
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure data_aux has the same number of features as expected by the model
            while len(data_aux) < NUM_EXPECTED_FEATURES:
                data_aux.append(0.0)

            # Truncate or pad data_aux if it exceeds the expected number of features
            data_aux = data_aux[:NUM_EXPECTED_FEATURES]

            # Get the predicted class directly
            predicted_class = model.predict([np.asarray(data_aux)])[0]

            # Display the predicted class on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 5
            text_size = cv2.getTextSize(predicted_class, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = 50
            cv2.putText(image, predicted_class, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # Create a Streamlit audio element for text-to-speech
            if predicted_class == 'TALK':
                tts = gTTS("Predicted: " + predicted_class)
                st.audio(tts.get_urls()[0])

    # Display the image with detected hand gestures
    st.image(image, channels="BGR", use_column_width=True)


footer="""<style>
header {visibility: hidden;}

/* Light mode styles */
p {
  color: black;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
  p {
    color: white;
  }
}

a:link , a:visited{
color: #5C5CFF;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

:root {
  --footer-bg-color: #333;
}

@media (prefers-color-scheme: dark) {
  :root {
    --footer-bg-color: rgb(14, 17, 23);
  }
}

@media (prefers-color-scheme: light) {
  :root {
    --footer-bg-color: white;
  }
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
font-size: 30rem;
background-color: var(--footer-bg-color);
color: black;
text-align: center;
}

</style>
<div class="footer">
<p>&copy; 2023 <a href="https://aliabdallah7.github.io/My-Portfolio/"> Ali Abdallah</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)