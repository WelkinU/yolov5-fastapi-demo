# yolov5-fastapi-demo

This is a demo FastAPI app that allows a user to upload an image, perform inference from a pretrained YOLOv5 model, and receive results in JSON format. This repo also includes Jinja2 HTML templates, so you can access this interface through a web browser at localhost:8000

Based on YOLOv5: https://github.com/ultralytics/yolov5

## Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7 (per https://github.com/ultralytics/yolov5). To install run:
```
pip install -r requirements.txt
```