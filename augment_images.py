import cv2
import numpy as np
import os

def augment_image(image):
    """
    Perform random rotations, flips, and brightness changes for data augmentation.
    """
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    alpha = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=alpha)
    return image

def augment_and_save(image_folder, output_folder):
    """
    Augment and save images from a folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        augmented_img = augment_image(img)
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, augmented_img)
    print(f"Augmented images saved to {output_folder}")

# Example usage
if __name__ == "__main__":
    image_folder = "segmented_images"
    output_folder = "augmented_images"
    augment_and_save(image_folder, output_folder)
