# Real-Time-Object-Detection
This repository contains a Flask web application that performs real-time object detection using a TensorFlow pre-trained model. The application captures video from a webcam, detects objects, and overlays bounding boxes and labels in real time.

Here's a structured directory layout for your Flask + TensorFlow Object Detection project,
object-detection-flask/
│── saved_model/                # Pre-trained TensorFlow model (Download separately)
│   └── (TensorFlow SavedModel files)
│── static/                     # Static files like CSS, JS, and images
│   ├── css/
│   │   └── styles.css           # Optional CSS for styling
│   ├── js/
│   │   └── script.js            # Optional JavaScript files
│── templates/                   # HTML templates for Flask
│   └── index.html               # Main webpage for the live video stream
│── app.py                       # Main Flask application
