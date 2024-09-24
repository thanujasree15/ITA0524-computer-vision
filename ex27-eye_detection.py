import cv2
img = cv2.imread(r'face_img2.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
haar_cascade = cv2.CascadeClassifier('Haarcascade_eye.xml')
faces = haar_cascade.detectMultiScale(gray_img , 1.1 , 15)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y) , (x+w,y+h) , (0,255,0) , 2)
cv2.imshow('Detected eye',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
