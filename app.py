# Import necessary libraries
from PIL import Image  # Import 'Image' from the PIL library for image handling
import cv2  # Import OpenCV for image manipulation
import mediapipe as mp  # Import the 'mediapipe' library for hand tracking
import pickle  # Import 'pickle' for data deserialization
import numpy as np  # Import 'numpy' for numerical operations
import streamlit as st  # Import Streamlit for creating web apps
from streamlit_lottie import st_lottie  # Import 'st_lottie' for displaying Lottie animations

# Open and display an image using Streamlit as a page icon
image = Image.open('./img/sign-language.png')  # Load an image from the file
st.set_page_config(page_title='ESLR', page_icon=image)  # Set page configuration with the image

# Set the language for the application
language = 'en'

# Load the trained model from the 'model.p' file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


# This module contains utility functions that allow you to draw landmarks and connections on images or frames.
# By assigning it to the variable mp_drawing,
# you can conveniently use these functions later in your code to visualize hand landmarks.
mp_drawing = mp.solutions.drawing_utils

# This module contains predefined styles for drawing landmarks and connections,
# which can be useful for customizing the visual appearance of hand landmarks
# when using the mp_drawing.draw_landmarks() function.
# Assigning it to the variable mp_drawing_styles allows you to access these styles in your code.
mp_drawing_styles = mp.solutions.drawing_styles


char_List = " "  # Initialize a character list
cap = cv2.VideoCapture(0)  # Initialize a video capture object for the webcam

stop_webcam = False  # Flag to indicate whether to stop the webcam

def main():
    global stop_webcam  # Use the global flag

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam

        if stop_webcam:  # Check the flag to stop the webcam
            break

        if not ret:
            st.write("Error: Unable to access the webcam.")
            break

        H, W, _ = frame.shape  # Get the height and width of the frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space

        results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # Image to draw on
                    hand_landmarks,  # Model output
                    mp_hands.HAND_CONNECTIONS,  # Hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
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
            while len(data_aux) < 100:
                data_aux.append(0.0)

            # Truncate or pad data_aux if it exceeds the expected number of features
            data_aux = data_aux[:100]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Get the predicted character directly from the model
            predicted_character = model.predict([np.asarray(data_aux)])[0]

            # Draw a rectangle and the predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # Set the size of the OpenCV window
        cv2.namedWindow('Webcam Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam Window', 1000, 750)  # Set the larger frame size

        cv2.imshow('Webcam Window', frame)  # Display the frame
        key = cv2.waitKey(1)  # Wait for a key press event

        if key == ord('q'):
            stop_webcam = True  # Set the flag to stop the webcam
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Create a Streamlit web app title
st.title("Sign Language Detector")

st.write("Please Click 'q' to quit and shut down the camer")

# Create a button to start the webcam
if st.button("Start Webcam"):
    main()  # Call the main function to start the webcam and sign language detection



style = """<style>
div.stButton > button {
  position: relative;
  display: flex;
  justify-content: space-between;
  width: 34.5rem;
  height: 5rem;
}
div.stButton > button:first-child {
  position: relative;
  display: inline-flex;
  justify-content: center;
  align-items: center;
  width: 15rem;
  height: 100%;
  background: var(--main-color);
  border: .2rem solid var(--main-color);
  border-radius: .8rem;
  font-size: 1.8rem;
  font-weight: 600;
  letter-spacing: .1rem;
  color: var(--bg-color);
  z-index: 1;
  overflow: hidden;
  transition: .5s;
}

div.stButton > button:hover:first-child {
   color: var(--main-color);
}

div.stButton > button:first-child::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 0;
  height: 100%;
  background: var(--bg-color);
  z-index: -1;
  transition: .5s;
}
div.stButton > button:hover::before {
  width: 100%;
}
button.style.border = '2px solid var(--main-color)';
header {visibility: hidden;}

/* Light mode styles */
.my-paragraph {
  color: black;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
  .my-paragraph {
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
  --bg-color: #081b29;
  --second-bg-color: #112e42;
  --text-color: #ededed;
  --main-color: #00abf0;
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
background-color: var(--footer-bg-color);
color: black;
text-align: center;
}

</style>


<div class="footer">
<p class="my-paragraph">&copy; 2023 <a href="https://www.linkedin.com/in/ali-abdallah7/"> Ali Abdallah</a></p>
</div>
"""


st.markdown(style,unsafe_allow_html=True)