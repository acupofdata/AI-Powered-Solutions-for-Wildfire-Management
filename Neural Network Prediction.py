pythonCopy codeimport tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and Preprocess Dataset
data = pd.read_csv('satellite_images.csv')
X = data.drop('fire_probability', axis=1)
y = data['fire_probability']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Simple Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {accuracy:.2f}")