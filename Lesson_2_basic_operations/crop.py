#crop 

import os
import cv2

image = cv2.imread(os.path.join('.','shake.jpg'))

print(image.shape)

cropped_image = image[45:135, 0: 249]

cv2.imshow('image', image)
cv2.imshow('cropped image', cropped_image)
cv2.waitKey(0)
