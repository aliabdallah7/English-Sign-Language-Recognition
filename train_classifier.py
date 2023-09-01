import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract the data and labels
data = data_dict.get('data', [])
labels = data_dict.get('labels', [])

# Define the maximum sequence length (adjust this as needed)
max_sequence_length = 100  # Example: Set to an appropriate value

# Pad or truncate sequences to the fixed length
data_padded = pad_sequences(data, maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')

# Convert the padded data to a NumPy array
data_array = np.array(data_padded)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
