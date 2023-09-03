# Import necessary libraries
import os  # Import the 'os' module for file and directory operations
import pickle  # Import 'pickle' for data serialization
import mediapipe as mp  # Import the 'mediapipe' library for hand tracking
import cv2  # Import OpenCV for image manipulation
import numpy as np  # Import 'numpy' for numerical operations
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier for machine learning
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
from sklearn.metrics import accuracy_score  # Import accuracy_score for evaluating model performance

# Define constants
DATA_DIR = 'data/train'  # Set the directory containing training data

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # Create an instance of the hands module from mediapipe
mp_drawing = mp.solutions.drawing_utils  # Utility functions for drawing landmarks on images
mp_drawing_styles = mp.solutions.drawing_styles  # Styles for drawing landmarks

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Create a hands detection model

# Initialize data lists
data = []  # Create an empty list to store hand landmark data
labels = []  # Create an empty list to store labels (directory names)

# Loop through the directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through image files in each directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Create an empty list to store landmark data for a single image
        x_ = []  # Create empty lists to store x-coordinates of landmarks
        y_ = []  # Create empty lists to store y-coordinates of landmarks
        
        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Convert the image to RGB color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Loop through detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through all landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # Get y-coordinate of the landmark
                    x_.append(x)  # Append x-coordinate to the list
                    y_.append(y)  # Append y-coordinate to the list
                
                # Calculate the relative coordinates of landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Calculate relative x-coordinate
                    data_aux.append(y - min(y_))  # Calculate relative y-coordinate
            
            # Append the landmark data and corresponding label to the data and labels lists
            data.append(data_aux)
            labels.append(dir_)

# Open a binary file to save the data and labels using pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)  # Serialize and save data and labels
f.close()  # Close the file
