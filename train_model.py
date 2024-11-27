import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from extract_features import compute_shape_features

# Load the dataset
df = pd.read_csv("extracted_features.csv")
X = df.iloc[:, 1:].values  # 2D features (Width, Height, etc.)
y = np.random.rand(len(X), 3)  # Replace with actual 3D measurements (manually collected data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # First hidden layer with 128 units
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting (50% dropout rate)
model.add(Dense(256, activation='relu'))  # Second hidden layer with 256 units
model.add(Dense(3, activation='linear'))  # Output layer predicting 3D measurements

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Save the model after training
model.save('trained_model.keras')

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")
