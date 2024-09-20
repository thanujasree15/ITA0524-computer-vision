import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the absolute values of the gradients
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    return sobel_combined

image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')

sobel_edges = sobel_edge_detection(image)

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_edges, cmap='gray')
plt.axis('off')

plt.show()
