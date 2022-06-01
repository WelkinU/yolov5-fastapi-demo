'''
This file is a barebones FastAPI example that:
  1. Accepts GET request, renders a HTML form at localhost:8000 allowing the user to
     upload a image and select YOLO model, then submit that data via POST
  2. Accept POST request, run YOLO model on input image, return JSON output

Works with client_minimal.py

This script does not require any of the HTML templates in /templates or other code in this repo
and does not involve stuff like Bootstrap, Javascript, JQuery, etc.
'''

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse

from PIL import Image
from io import BytesIO

import torch

app = FastAPI()

@app.get("/")
async def home(request: Request):
  ''' Returns barebones HTML form allowing the user to select a file and model '''

  html_content = '''
<form method="post" enctype="multipart/form-data">
  <div>
    <label>Upload Image</label>
    <input name="file" type="file" multiple>
    <div>
      <label>Select YOLO Model</label>
      <select name="model_name">
        <option>yolov5s</option>
        <option>yolov5m</option>
        <option>yolov5l</option>
        <option>yolov5x</option>
      </select>
    </div>
  </div>
  <button type="submit">Submit</button>
</form>
'''

  return HTMLResponse(content=html_content, status_code=200)


@app.post("/")
async def process_home_form(file: UploadFile = File(...), 
              model_name: str = Form(...)):
  
    '''
    Requires an image file upload, model name (ex. yolov5s).
    Returns: json response with list of list of dicts.
      Each dict contains class, class_name, confidence, normalized_bbox

    Note: Because this is an async method, the YOLO inference is a blocking
    operation.
    '''

    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload = False)

    #This is how you decode + process image with PIL
    results = model(Image.open(BytesIO(await file.read())))

    #This is how you decode + process image with OpenCV + numpy
    #results = model(cv2.cvtColor(cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))

    json_results = results_to_json(results,model)
    return json_results


def results_to_json(results, model):
    ''' Helper function for process_home_form()'''
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


if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server_minimal:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)