import os
import cv2

img = cv2.imread(os.path.join('.','birds.jpg'))

#Threshold, into White for detection
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21 ,30)
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    #prints out the contour sizes, or the area of each bird
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt) > 200:
        #This works for the orginal image
        # cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)

        #Makes a box around the contours 
        x, y, w, h = cv2.boundingRect(cnt)

        # Based off the Thresh we apply the boxes onto the main image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', img)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)