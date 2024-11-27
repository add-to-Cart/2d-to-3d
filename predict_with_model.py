import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from augment_images import augment_image
from extract_features import compute_shape_features
from mpl_toolkits.mplot3d import Axes3D

# Load the trained model
model = load_model('trained_model.keras')

# Image prediction (predicting 3D measurements from a new image)
new_image_path = "images/william.png"
new_image = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if new_image is None:
    print(f"Error: Image at {new_image_path} could not be loaded.")
else:
    print("Image loaded successfully.")

    # Apply augmentation
    augmented_new_image = augment_image(new_image)

    # Extract features from the augmented image (no need to pass a path)
    features_new = compute_shape_features(augmented_new_image)  # Pass the image directly

    # Ensure that features_new is a 2D array (model expects a batch of samples)
    features_new = np.expand_dims(features_new, axis=0)  # Reshape to (1, n_features)

    # Predict the 3D measurements
    predicted_3d_measurements = model.predict(features_new)
    print(f"Predicted 3D measurements: {predicted_3d_measurements}")

    # Assuming the model outputs a 3D coordinate, extract x, y, z
    x, y, z = predicted_3d_measurements[0]

    # Visualizing the predicted 3D measurements using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the predicted 3D point
    ax.scatter(x, y, z, color='r', label='Predicted 3D Point')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a title and a legend
    ax.set_title('Predicted 3D Measurements')
    ax.legend()

    # Show the plot
    plt.show()
