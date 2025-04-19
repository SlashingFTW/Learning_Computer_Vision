import boto3
import cv2
import credentials
import os

output_dir = './data2'
os.makedirs(output_dir, exist_ok=True)
video_output_path = os.path.join(output_dir, 'webcam_output_fast.mp4')

# AWS Rekognition setup
reko_client = boto3.client('rekognition',
    aws_access_key_id=credentials.access_key,
    aws_secret_access_key=credentials.secret_key,
    region_name='us-east-2'
)

# Set target class
target_class = 'Can'

# Open webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to access webcam.")
    exit()

H, W, _ = frame.shape
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (W, H))

print("Press 'q' to stop")

frame_num = 0
prev_boxes = []  # store previous detection boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only run Rekognition every 5 frames
    if frame_num % 5 == 0:
        # Resize frame before sending (for speed)
        small_frame = cv2.resize(frame, (320, 240))
        small_H, small_W, _ = small_frame.shape

        _, buffer = cv2.imencode('.jpg', small_frame)
        image_bytes = buffer.tobytes()

        response = reko_client.detect_labels(Image={'Bytes': image_bytes}, MinConfidence=50)

        prev_boxes = []  # reset boxes
        for label in response['Labels']:
            if label['Name'] == target_class:
                for instance in label['Instances']:
                    bbox = instance['BoundingBox']
                    x1 = int(bbox['Left'] * W)
                    y1 = int(bbox['Top'] * H)
                    width = int(bbox['Width'] * W)
                    height = int(bbox['Height'] * H)
                    prev_boxes.append((x1, y1, width, height, label['Name']))

    # Draw previous detection boxes
    for (x1, y1, width, height, name) in prev_boxes:
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show and save
    cv2.imshow('Live Detection (Fast)', frame)
    out.write(frame)

    print(f"Frame {frame_num}", end='\r')
    frame_num += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nSaved video: {video_output_path}")
