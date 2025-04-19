import cv2
import os

#read image 
image_path = os.path.join('.', 'data', 'bird.jpg')
bird_im = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), bird_im)

# visualize image

cv2.imshow('Bird', bird_im)
cv2.waitkey(0)