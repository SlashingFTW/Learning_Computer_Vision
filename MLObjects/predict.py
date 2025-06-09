import os
from ultralytics import YOLO
import cv2
import math

VIDEOS_DIR = os.path.join('.', 'test_dummy')
video_path = os.path.join(VIDEOS_DIR, 'cans.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret or frame is None:
    print("❌ Failed to read the video.")
    exit()

H, W, _ = frame.shape
frame_area = W * H

out = cv2.VideoWriter(
    video_path_out,
    cv2.VideoWriter_fourcc(*'mp4v'),  # use 'mp4v' for compatibility
    int(cap.get(cv2.CAP_PROP_FPS)),
    (W, H)
)

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Calculate box area
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Skip boxes that are too large
            if box_area > 0.8 * frame_area:
                continue

            # Get center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Format label and confidence
            label = model.names[int(class_id)].upper()
            confidence = math.ceil(score * 100 / 5) * 5  # Round up to nearest 5%
            label_text = f"{label} {confidence}%"

            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, label_text, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Draw center dot
            cv2.circle(frame, (center_x, center_y), 6, (255, 0, 0), -1)

            # Draw center coordinates
            center_text = f"({center_x}, {center_y})"
            cv2.putText(frame, center_text, (center_x + 10, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Optional: print to terminal
            print(f"Detected {label} {confidence}% at center: {center_text}")

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
