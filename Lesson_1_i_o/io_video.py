import cv2
import os

#read video
video_path = os.path.join('.', 'data', 'cars.mp4')
video = cv2.VideoCapture(video_path)

#visualize video 

ret = True
while ret:
    ret, frame = video.read()
    # ret == the amount of frames in the video 
    # frame == current frame
    if ret: 
        cv2.imshow('frame', frame)
        cv2.waitKey(40) # 1 sec/(frame rate) = waitKey : waitKey is the amount of time it takes before looking at a new frame

video.release()
cv2.destroyAllWindows()