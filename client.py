''' Example client sending POST request to server (localhost:8000/detect/)and printing the YOLO results

The send_request() function has a couple options demonstrating all the ways you can interact 
with the /detect endpoint
'''

import requests as r
import json
from pprint import pprint

import base64
from io import BytesIO
from PIL import Image

def send_request(file_list = ['./images/zidane.jpg'], 
                    model_name = 'yolov5s',
                    img_size = 640,
                    download_image = False):

    #upload multiple files as list of tuples
    files = [('file_list', open(file,"rb")) for file in file_list]

    #pass the other form data here
    other_form_data = {'model_name': model_name,
                    'img_size': img_size,
                    'download_image': download_image}

    res = r.post("http://localhost:8000/detect/", 
                    data = other_form_data, 
                    files = files)

    if download_image:
        json_data = res.json()

        for img_data in json_data:
            for bbox_data in img_data:
                #parse json to detect if the dict contains image data (base64) or bbox data
                if 'image_base64' in bbox_data.keys():
                    #decode and show base64 encoded image
                    img = Image.open(BytesIO(base64.b64decode(str(bbox_data['image_base64']))))
                    img.show()
                else:
                    #otherwise print json bbox data
                    pprint(bbox_data)

    else:
        #if no images were downloaded, just display json response
        pprint(json.loads(res.text))


if __name__ == '__main__':
    #example uploading image batch
    #send_request(file_list=['./images/bus.jpg','./images/zidane.jpg'])

    #example uploading image and receiving bbox json + image with bboxes drawn
    send_request(file_list=['./images/bus.jpg'], download_image = True)
