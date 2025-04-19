import os 
import cv2

img = cv2.imread(os.path.join('.','handwritten.png'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#extra Research, allows the image to broekn down into parts allowing for each section to be based off its own threshold 
thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21 ,30)

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)