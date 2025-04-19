from ultralytics import YOLO
import cv2

#load yolov8
model = YOLO('yolov8n.pt')

#load video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

#read Video
while ret:
    ret, frame = cap.read()

    if ret:
        #object detection
        #track objects
        results = model.track(frame, persist = True) #Track all objects through each frame and remember those objects

        #plot results
        # could use cv2.plotText and cv2.rectangle
        frame_ = results[0].plot()

        #visualize 
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 