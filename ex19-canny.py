import cv2
import matplotlib.pyplot as plt

def canny_edge_detection(image, low_threshold, high_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges


image = cv2.imread(r'D:\shifted_C\New folder\nature1.jpeg')
canny_edges = canny_edge_detection(image, 100, 200)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edge Detection')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.show()
