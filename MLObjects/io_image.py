import cv2
import os

#read image 
image_path = os.path.join('.', 'test_dummy', 'cokecans.jpeg')
img = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('.', 'test_dummy', 'cokecans.jpeg'), img)

# visualize image

cv2.imshow('Image', img)
cv2.waitKey(0)