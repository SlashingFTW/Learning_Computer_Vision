import boto3
import cv2
import credentials
import os

output_dir = './data2'
output_dir_anns = os.path.join(output_dir, 'anns')

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_anns, exist_ok=True)

# Create AWS Rekognition client
reko_client = boto3.client('rekognition',
    aws_access_key_id=credentials.access_key,
    aws_secret_access_key=credentials.secret_key,
    region_name='us-east-2'
)

# Set the target class
target_class = 'Can'

# Load the video
cap = cv2.VideoCapture('./cans.mp4')

# Get frame dimensions and FPS
ret, first_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    exit()

H, W, _ = first_frame.shape
fps = 30  # Set to fixed FPS as requested
video_output_path = os.path.join(output_dir, 'output.mp4')

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (W, H))

# Process first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_num = -1
ret = True

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}", end='\r')
    H, W, _ = frame.shape
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Detect objects
    response = reko_client.detect_labels(
        Image={'Bytes': image_bytes},
        MinConfidence=50
    )

    with open(os.path.join(output_dir_anns, f'frame{str(frame_num).zfill(6)}.txt'), 'w') as f:
        for label in response['Labels']:
            if label['Name'] == target_class:
                for instance in label['Instances']:
                    bbox = instance['BoundingBox']
                    x1 = int(bbox['Left'] * W)
                    y1 = int(bbox['Top'] * H)
                    width = int(bbox['Width'] * W)
                    height = int(bbox['Height'] * H)

                    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)

                    f.write('{} {} {} {} {}\n'.format(
                        0,
                        x1 + width / 2,
                        y1 + height / 2,
                        width,
                        height
                    ))

    # Write frame to video
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f"Video saved as {video_output_path}")
