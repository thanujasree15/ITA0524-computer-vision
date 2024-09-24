import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_top_hat(image, kernel_size=(15, 15)):
    """
    Applies the Top-Hat morphological operation to an input image.
    
    Parameters:
    - image: The input image (grayscale or color).
    - kernel_size: The size of the structuring element.
    
    Returns:
    - top_hat: The image after applying the Top-Hat transformation.
    """
    # Convert image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Create a structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply the Top-Hat transformation
    top_hat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
    
    return top_hat

# Load the input image
image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')

# Apply Top-Hat operation
top_hat_image = apply_top_hat(image, kernel_size=(15, 15))

# Display the original and Top-Hat transformed images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Top-Hat Transformed Image
plt.subplot(1, 2, 2)
plt.title('Top-Hat Image')
plt.imshow(top_hat_image, cmap='gray')
plt.axis('off')

plt.show()
