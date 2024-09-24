import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_threshold(image_path, threshold_value=128, max_value=255):
    """
    Applies binary thresholding to an input image.
    
    Parameters:
    - image_path: Path to the input image.
    - threshold_value: The threshold value to segment the image.
    - max_value: The maximum value to set for pixels exceeding the threshold.
    
    Returns:
    - thresholded_image: The image after applying binary thresholding.
    """
    # Load the input image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Apply binary thresholding
    retval, thresholded_image = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY)
    
    return image, thresholded_image

# Path to the image
image_path = r'D:\shifted_C\New folder\nature1.jpeg'

# Perform binary thresholding
original_image, thresholded_image = binary_threshold(image_path, threshold_value=128, max_value=255)

# Display the original and thresholded images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

# Thresholded Image
plt.subplot(1, 2, 2)
plt.title('Binary Thresholded Image')
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')

plt.show()
