import os 
import cv2

img = cv2.imread(os.path.join('.','birds.jpeg'))

#1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#2)
# 80 is the bottom threshold otherwise go to 255
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('thresh', thresh)
#3) Optional - Makes the binary look more distinct 
# thresh = cv2.blur(thresh,(10,10))
# ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('img gray', img_gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)