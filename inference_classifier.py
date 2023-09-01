from tensorflow.keras.models import load_model
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound  

language = 'en'
model_dict = pickle.load(open('./model.p', 'rb'))
# model = load_model('model.h5')
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the number of expected features (adjust this based on your model)
num_expected_features = 100

char_List = " "
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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
        while len(data_aux) < num_expected_features:
            data_aux.append(0.0)

        # Truncate or pad data_aux if it exceeds the expected number of features
        data_aux = data_aux[:num_expected_features]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Get the predicted character directly
        predicted_character = model.predict([np.asarray(data_aux)])[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA) 
        if predicted_character != char_List[-1] and predicted_character != 'T':
            char_List += predicted_character
            print(char_List)
        if predicted_character == 'T':
            obj = gTTS(char_List)
            obj.save("speech.mp3")
            break
            
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
playsound("speech.mp3")
