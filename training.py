# Import necessary libraries
import pickle  # Import 'pickle' for data deserialization
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier for machine learning
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
from sklearn.metrics import accuracy_score  # Import accuracy_score for evaluating model performance
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences for sequence padding
import numpy as np  # Import 'numpy' for numerical operations

# Load the previously saved data from the 'data.pickle' file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels from the loaded data dictionary
data = data_dict.get('data', [])  # Retrieve the 'data' key or an empty list if not found
labels = data_dict.get('labels', [])  # Retrieve the 'labels' key or an empty list if not found

# Define the maximum sequence length for sequence padding (adjust as needed)
max_sequence_length = 100  # Example: Set to an appropriate value

# Pad or truncate sequences to the fixed length using 'pad_sequences'
data_padded = pad_sequences(data, maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')

# Convert the padded data to a NumPy array
data_array = np.array(data_padded)

# Split the data into training and testing sets using 'train_test_split'
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_predict = model.predict(x_test)

# Calculate the accuracy score of the model's predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy score as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a binary file using 'pickle'
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)  # Serialize and save the model
f.close()  # Close the file