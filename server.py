from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                        'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

##############################################
#-------------GET Request Routes--------------
##############################################
@app.get("/")
def home(request: Request):
    ''' Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
            "request": request,
            "model_selection_options": model_selection_options,
        })

@app.get("/drag_and_drop_detect")
def drag_and_drop_detect(request: Request):
    ''' drag_and_drop_detect detect page. Uses a drag and drop
    file interface to upload files to the server, then renders 
    the image + bboxes + labels on HTML canvas.
    '''

    return templates.TemplateResponse('drag_and_drop_detect.html', 
            {"request": request,
            "model_selection_options": model_selection_options,
        })


##############################################
#------------POST Request Routes--------------
##############################################
@app.post("/")
def detect_with_server_side_rendering(request: Request,
                        file_list: List[UploadFile] = File(...), 
                        model_name: str = Form(...),
                        img_size: int = Form(640)):
    
    '''
    Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).

    Returns: HTML template render showing bbox data and base64 encoded image

    Notes: 
    Intended to show how to do server sided image rendering + passing to client. But
    generally, you will just want to return results as JSON and do the rendering client side.
    See templates/drag_and_drop_detect.html for an example on how to do this.

    If you just want JSON results, just return the results of the 
    results_to_json() function and skip the rest
    '''

    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                    for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model_dict[model_name](img_batch_rgb, size = img_size)

    json_results = results_to_json(results,model_dict[model_name])

    img_str_list = []
    #plot bboxes on the image
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label, 
                    color=colors[int(bbox['class'])], line_thickness=3)

        img_str_list.append(base64EncodeImage(img))

    #escape the apostrophes in the json string representation
    encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

    return templates.TemplateResponse('show_results.html', {
            'request': request,
            'bbox_image_data_zipped': zip(img_str_list,json_results), #unzipped in jinja2 template
            'bbox_data_str': encoded_json_results,
        })


@app.post("/detect")
def detect_via_api(request: Request,
                file_list: List[UploadFile] = File(...), 
                model_name: str = Form(...),
                img_size: Optional[int] = Form(640),
                download_image: Optional[bool] = Form(False)):
    
    '''
    Requires an image file upload, model name (ex. yolov5s). 
    Optional image size parameter (Default 640)
    Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response
    
    Returns: JSON results of running YOLOv5 on the uploaded image. Bbox format is X1Y1X2Y2. 
            If download_image parameter is True, images with
            bboxes drawn are base64 encoded and returned inside the json response.

    Intended for API usage.
    '''

    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    
    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB, 
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    
    results = model_dict[model_name](img_batch_rgb, size = img_size) 
    json_results = results_to_json(results,model_dict[model_name])

    if download_image:
        #server side render the image with bounding boxes
        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['bbox'], img, label=label, 
                        color=colors[int(bbox['class'])], line_thickness=3)

            payload = {'image_base64':base64EncodeImage(img)}
            json_results[idx].append(payload)

    encoded_json_results = str(json_results).replace("'",r'"')
    return encoded_json_results
    
##############################################
#--------------Helper Functions---------------
##############################################

def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def base64EncodeImage(img):
    ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64

if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default = 'localhost')
    parser.add_argument('--port', default = 8000)
    parser.add_argument('--precache-models', action='store_true', 
            help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    if opt.precache_models:
        model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True) 
                        for model_name in model_selection_options}
    
    app_str = 'server:app' #make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host= opt.host, port=opt.port, reload=True)
