import cv2

image = cv2.imread(r"D:\shifted_C\New folder\nature1.jpeg", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_colored, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', image_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
