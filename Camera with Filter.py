import cv2
import numpy as np
# face recognition
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# captures the image
cap=cv2.VideoCapture(0)

# on pressing the key 'C' the image gets captured
# to add filters :
# key press 'B' will put a blue filter
# key press 'G' will put a green filter
# key press 'R' will put a red filter
# image will be saved as a jpg file in the same location the project is launched
while cap.isOpened():
    _, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=11)

    for (x,y,w,h) in f:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
    cv2.imshow('img',img)

    if cv2.waitKey(1) == ord('c'):
        break

_, img=cap.read()
cv2.imshow('img',img)
cv2.imwrite('pic.jpg',img)
img=cv2.imread('pic.jpg')
B,G,R=cv2.split(img)
zeros=np.zeros(img.shape[:2],dtype="uint8")
if cv2.waitKey(0) == ord('b'):
    cv2.imshow('filter', cv2.merge([B, zeros, zeros]))
if cv2.waitKey(0) == ord('g'):
    cv2.imshow('filter', cv2.merge([zeros, G, zeros]))
if cv2.waitKey(0) == ord('r'):
    cv2.imshow('filter', cv2.merge([zeros, zeros, R]))

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
