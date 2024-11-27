import cv2
import numpy as np
import os
import pandas as pd

# In extract_features.py

def compute_shape_features(image):
    """
    Compute 2D features like width, height, area, perimeter, aspect ratio, compactness, and solidity.
    """
    # Check if input is an image path (string) or an image array (NumPy array)
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        img = image  # If it's already an array, use it directly
    else:
        raise ValueError("Input should be either a file path or a NumPy array.")

    # Apply binary threshold to get a binary image
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        aspect_ratio = width / float(height) if height != 0 else 0
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        return [width, height, area, perimeter, aspect_ratio, compactness, solidity]
    return []


def extract_features(image_folder, output_csv):
    """
    Extract 2D features from images in a folder and save to a CSV file.
    """
    data = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        features = compute_shape_features(img_path)
        if features:
            data.append([img_name] + features)
    columns = ["Image", "Width", "Height", "Area", "Perimeter", "Aspect_Ratio", "Compactness", "Solidity"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    image_folder = "augmented_images"
    output_csv = "extracted_features.csv"
    extract_features(image_folder, output_csv)
