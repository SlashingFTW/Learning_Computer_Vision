import boto3
import cv2
import credentials
import os

output_dir = './data2' #Move outputs to data
output_dir_imgs = os.path.join(output_dir, 'imgs') #Move images to imgs
output_dir_anns = os.path.join(output_dir, 'anns') #Move anns to anns

# create AWS Reko Client
reko_client = boto3.client('rekognition', aws_access_key_id=credentials.access_key,aws_secret_access_key=credentials.secret_key, region_name= 'us-east-2')

# set the traget class
target_class = 'Can'

# load the video 
cap = cv2.VideoCapture('./cans.mp4')

frame_num = -1

# read frames
ret = True
while ret:
    ret, frame = cap.read()

    if ret:
        frame_num += 1
        H, W, _ = frame.shape
        #convert o jpg
        _, buffer = cv2.imencode('.jpg', frame)

        #convert buffer to bytes
        image_bytes = buffer.tobytes()

        #detect objects
        response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                MinConfidence=50)
        
        with open(os.path.join(output_dir_anns, 'frame{}.txt'.format(str(frame_num).zfill(6))), 'w') as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instance_num in range(len(label['Instances'])):
                        bounding_box = label['Instances'][instance_num]['BoundingBox']
                        x1 = int(bounding_box['Left'] * W)
                        y1 = int(bounding_box['Top'] * H)
                        width = int(bounding_box['Width'] * W)
                        height = int(bounding_box['Height'] * H)

                        cv2.rectangle(frame, (x1,y1), (x1 + width, y1 +height), (0, 255, 0), 3)
            
                        #write detections 
                        f.write('{}{}{}{}{}\n'.format(0,
                                                    (x1 + width / 2),
                                                    (y1 + height /2),
                                                    width,
                                                    height)
                                )
            f.close()

        cv2.imwrite(os.path.join(output_dir_imgs, 'frame{}.jpg'.format(str(frame_num).zfill(6))), frame)
        