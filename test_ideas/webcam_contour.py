import cv2

# Load the face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert frame to grayscale (for face detection and Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles on the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show both the edge detection and face detection
    cv2.imshow("Face Tracking", frame)
    cv2.imshow("Canny Edges", edges)

    # Press 'q' to exit
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
