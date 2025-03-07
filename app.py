# app.py
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the pre-trained model from TensorFlow Hub
model = model = tf.saved_model.load("saved_model")

# Full COCO category labels
category_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

def preprocess_frame(frame):
    input_tensor = tf.image.convert_image_dtype(frame, dtype=tf.uint8)[tf.newaxis, ...]
    detections = model(input_tensor)  # Run detection
    return detections

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run object detection
        detections = preprocess_frame(frame)

        # Extract bounding box and label data from detections
        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(int)
        scores = detections["detection_scores"][0].numpy()

        for i in range(len(scores)):
            if scores[i] > 0.5:  # Display boxes above a confidence threshold
                box = boxes[i]
                y_min, x_min, y_max, x_max = box
                label = category_index.get(classes[i], "N/A")
                frame = cv2.rectangle(
                    frame, 
                    (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                    (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])),
                    (0, 255, 0), 2
                )
                frame = cv2.putText(
                    frame, label, 
                    (int(x_min * frame.shape[1]), int(y_min * frame.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
