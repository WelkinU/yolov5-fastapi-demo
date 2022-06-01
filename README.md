# yolov5-fastapi-demo

This is a demo FastAPI app that allows a user to upload image(s), perform inference using a pretrained YOLOv5 model, and receive results in JSON format. This repo also includes Jinja2 HTML templates, so you can access this interface through a web browser at `localhost:8000`.

<img src="https://user-images.githubusercontent.com/47000850/171301696-fe31b6fd-a2c4-4b2c-9029-f11ce1ddfb64.png" alt="image" width="700"/>

## Install Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7 (per https://github.com/ultralytics/yolov5).

To install run:
```
pip install -r requirements.txt
```

## Minimal FastAPI Example

See the `minimal_client_server_example` folder for a minimal client/server wrapper of YOLOv5 with FastAPI and HTML forms.

Again, you can try this out by:
1. Running the server with `python server_minimal.py` or `uvicorn server_minimal:app --reload`
1. Test the server with `python client_minimal.py`, navigating to `localhost:8000` in your web browser or `localhost:8000/docs -> POST Request -> Try It Out`

## Inference Methods

You can initialize the server with `python server.py` (use `--help` for other args) or `uvicorn server:app --reload`

You can test the server a couple of ways:
1. Using `client.py` - this is a basic example of using the Requests library to upload a batch of images + model name to `localhost:8000/detect/` and receive JSON inference results. 
1. Open `localhost:8000` in your web browser, use the web form to upload image(s) and select a model, then click submit. You should see inference results displayed in the web browser shortly. 
1. Open `http://localhost:8000/drag_and_drop_detect` in your web browser, use the drag and drop interface to upload an image, and the image + bounding boxes will be rendered via Javascript.

Models will automatically be downloaded the first time they are used and are cached on disc.

<img src="https://user-images.githubusercontent.com/47000850/171300877-e3941e01-1aa0-4816-9cf9-6947481b4ec8.png" alt="city_street_results" width="700"/>

## API Documentation
API endpoint documentation is auto-generated in `localhost:8000/docs`. The general idea is that humans use the "/" route (HTML form + inference results displayed in the browser) and programs use the "/detect/" API route to receive JSON inference results.

## Developer Notes

### Server.py

Contains the FastAPI server code and helper functions.

### Jinja2 Templates (`/templates` folder)

| File | Description |
| --- | --- | 
| layout.html | Base template with navbar that is common to all pages. `home.html` and `drag_and_drop_detect.html` both extend this template. |
| `home.html` | Basic web form for uploading images, model selection and inference size to the server. The server gets the YOLO results and renders a bbox image, then returns the results by plugging them into the jinja2 template `templates/show_results.html`. This is overly fancy, but I wanted to demonstrate how to do this - if you want just JSON results see the minimal client-server example. |
| `drag_and_drop_detect.html` | This implements a Drag & Drop interface to upload images. Once dropped onto the dropzone, the image and parameters are sent to the server's `/detect` endpoint which returns JSON results. The JSON results are then used to render the image + bboxes in the web browser as seen in the Inference Methods section above. The box labels are raised above the box outline such that the labels don't overlap with each other. |
 
## Credits

This repository is a wrapper around YOLOv5 from Ultralytics: https://github.com/ultralytics/yolov5

Also modified the results_to_json function from the original here: https://gist.github.com/decent-engineer-decent-datascientist/81e04ad86e102eb083416e28150aa2a1
