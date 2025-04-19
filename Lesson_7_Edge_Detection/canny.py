import os 
import cv2 
import numpy as np

image = cv2.imread(os.path.join('.','cokecans.jpeg'))

image_edge = cv2.Canny(image, 200, 500)

image_dilate = cv2.dilate(image_edge, np.ones((5,5),dtype = np.int8))

image_erosion = cv2.erode(image_edge, np.ones((3,3), dtype=np.int8))

cv2.imshow('image', image)
cv2.imshow('image edge', image_edge)
cv2.imshow('image d', image_dilate)
cv2.imshow('image erode', image_erosion)
cv2.waitKey(0)
