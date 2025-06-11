#  Waste Detection using YOLOv8 + FastAPI

This project is a computer vision application that detects various types of waste (e.g., trash, cans, organic waste, paper, plastic bags) in images using the YOLOv8 object detection model. It features a modern, lightweight web interface powered by FastAPI, and allows users to upload images or use their webcam for real-time detection.

---

##  Features

- Upload or capture images for waste detection
- Real-time webcam support
- Visual feedback with bounding boxes
- Displays detected waste categories
- Clean, responsive frontend with HTML/CSS
- Easily deployable as a web app

---

##  Model Training

The model was trained using [Ultralytics YOLOv8](https://docs.ultralytics.com) on a custom annotated waste dataset consisting of several classes including:
- **Trash**
- **Organic Waste**
- **Plastic Bag**
- **Carton**
- **Paper**
- **Phone Case**
- **Can**

###  Training Details:
- Framework: YOLOv8 via Ultralytics
- Dataset: Custom dataset with images + YOLO-format annotations
- Training Script: Executed on Google Colab
- Output Model: `best.pt` exported from training directory

###  Evaluation Results:
- **Precision**: 0.699
- **Recall**: 0.834
- **mAP@0.5**: 0.848
- **mAP@0.5:0.95**: 0.785

---

## FastAPI Application

The FastAPI backend does the following:

- Serves a web interface (`index.html`)
- Handles image uploads or webcam captures
- Passes the image through the YOLOv8 model
- Returns predicted classes and annotated image with bounding boxes


