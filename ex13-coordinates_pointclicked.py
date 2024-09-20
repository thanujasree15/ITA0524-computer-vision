
import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Point clicked: ({x}, {y})')

image = cv2.imread(r"D:\shifted_C\New folder\nature1.jpeg")
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
