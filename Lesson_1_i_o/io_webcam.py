import cv2

# read Webcam
webcam = cv2.VideoCapture(0)

#vislize Webcam
while True:
    ret, frame =webcam.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'): #if 'q' is pressed it terminates 
        break

webcam.release()
cv2.destroyAllWindows()