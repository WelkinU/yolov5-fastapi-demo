# yolov5-fastapi-demo

This is a demo FastAPI app that allows a user to upload an image, perform inference using a pretrained YOLOv5 model, and receive results in JSON format. This repo also includes Jinja2 HTML templates, so you can access this interface through a web browser at localhost:8000

<img src="https://user-images.githubusercontent.com/47000850/107603157-e05aae00-6bf9-11eb-8c1a-2715fdc27066.png" alt="image" width="630"/>

<img src="https://user-images.githubusercontent.com/47000850/107911503-bae7e000-6f2a-11eb-8be4-cf662608546e.png" alt="image" width="630"/>


## Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7 (per https://github.com/ultralytics/yolov5).

To install run:
```
pip install -r requirements.txt
```

## Inference

You can initialize the server with `python server.py` or `uvicorn server:app --reload`

You can test the server using `client.py` or by opening `localhost:8000` in your web browser. You can test the POST requests by going to localhost:8000/docs, clicking on a request and then clicking the "Try It Out" Button.

Models will automatically be downloaded the first time they are used.

## Minimal FastAPI Example

See the `fastapi_yolov5_minimal_client_server_example` folder for a minimal client/server wrapper of YOLOv5 with FastAPI and HTML forms.

Again, you can try this out by:
1. Running the server with `python server_minimal.py` or `uvicorn server_minimal:app --reload`
1. Test the server with `python client_minimal.py`, navigating to `localhost:8000` in your web browser or `localhost:8000/docs -> POST Request -> Try It Out`

## Credits

This repository is a wrapper around YOLOv5 from Ultralytics: https://github.com/ultralytics/yolov5

Also grabbed some code/ideas from: https://gist.github.com/decent-engineer-decent-datascientist/81e04ad86e102eb083416e28150aa2a1
