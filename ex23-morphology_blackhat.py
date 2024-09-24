import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_black_hat(image, kernel_size=(15, 15)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    black_hat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    
    return black_hat

image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')
black_hat_image = apply_black_hat(image, kernel_size=(15, 15))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Black Hat Image')
plt.imshow(black_hat_image, cmap='gray')
plt.axis('off')

plt.show()
