import cv2
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    """
    Applies Roberts Cross edge detection to an input image.

    Parameters:
    - image: The input image.

    Returns:
    - roberts_combined: The result of combining Roberts gradients in the X and Y directions.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define Roberts Cross operator kernels
    roberts_cross_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_cross_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Apply the Roberts operator in the X direction
    edge_x = cv2.filter2D(gray_image, -1, roberts_cross_x)
    
    # Apply the Roberts operator in the Y direction
    edge_y = cv2.filter2D(gray_image, -1, roberts_cross_y)
    
    # Compute the absolute values of the gradients
    abs_edge_x = cv2.convertScaleAbs(edge_x)
    abs_edge_y = cv2.convertScaleAbs(edge_y)
    
    # Combine the Roberts X and Y gradients
    roberts_combined = cv2.addWeighted(abs_edge_x, 0.5, abs_edge_y, 0.5, 0)
    
    return roberts_combined

# Load the image
image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')

# Apply Roberts edge detection
roberts_edges = roberts_edge_detection(image)

# Display the original and Roberts edge-detected images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Roberts Edge Detection Result
plt.subplot(1, 2, 2)
plt.title('Roberts Edge Detection')
plt.imshow(roberts_edges, cmap='gray')
plt.axis('off')

plt.show()
