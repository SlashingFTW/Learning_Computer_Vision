#resizing 

import os
import cv2

image = cv2.imread(os.path.join('.','shake.jpg'))
resized_img = cv2.resize(image, (200,100))

print(image.shape)
print(resized_img.shape)


cv2.imshow('image',image)
cv2.imshow('resized image', resized_img)
cv2.waitKey(0)