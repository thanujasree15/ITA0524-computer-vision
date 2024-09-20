import cv2
import numpy as np
import matplotlib.pyplot as plt

def prewitt_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    edge_x = cv2.filter2D(gray_image, -1, prewitt_x)
    edge_y = cv2.filter2D(gray_image, -1, prewitt_y)
    abs_edge_x = cv2.convertScaleAbs(edge_x)
    abs_edge_y = cv2.convertScaleAbs(edge_y)
    prewitt_combined = cv2.addWeighted(abs_edge_x, 0.5, abs_edge_y, 0.5, 0)
    return prewitt_combined


image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')
prewitt_edges = prewitt_edge_detection(image)
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Prewitt Edge Detection Result
plt.subplot(1, 2, 2)
plt.title('Prewitt Edge Detection')
plt.imshow(prewitt_edges, cmap='gray')
plt.axis('off')

plt.show()
