import os
import cv2
from ultralytics import YOLO

# Set paths relative to MLObjects directory
video_input_path = os.path.join('test_dummy', 'cans.mp4')
video_output_path = os.path.join('test_dummy', 'cans_out.mp4')
model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')  # or 'last.pt'

output_dir_anns = os.path.join('test_dummy', 'anns')
os.makedirs(output_dir_anns, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_input_path)
ret, first_frame = cap.read()
if not ret:
    print("❌ Failed to read video.")
    exit()

H, W = first_frame.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Load YOLOv8 model
model = YOLO(model_path)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (W, H))

# Reset video and process
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_num = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}", end='\r')

    results = model(frame)[0]

    # Save annotations
    with open(os.path.join(output_dir_anns, f'frame{str(frame_num).zfill(6)}.txt'), 'w') as f:
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box

            # Only draw boxes for class_id 0 (Can) and above threshold
            if int(class_id) == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Save bounding box in YOLO-style center format (un-normalized)
                w = x2 - x1
                h = y2 - y1
                xc = x1 + w / 2
                yc = y1 + h / 2

                f.write(f'{int(class_id)} {xc} {yc} {w} {h}\n')

    out.write(frame)

# Cleanup
cap.release()
out.release()
print(f"\n✅ Done. Output video saved at: {video_output_path}")
