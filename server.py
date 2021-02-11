from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
from io import BytesIO
import torch

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

#so we can read main.css
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/")
def home(request: Request):
	'''
	Returns html jinja2 template render for home page
	'''

	return templates.TemplateResponse('home.html', {
			"request": request,
		})

def results_to_json(results, model):
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "normalized_box": pred[:4].tolist(),
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxyn
    ]

@app.post("/")
async def process_home_form(file: UploadFile = File(...), 
							model_name: str = Form(...)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s).
	Returns: json response with list of dicts. 
		Each dict contains class, class_name, confidence, normalized_bbox
	'''

	#NOTE: Can't use pydantic validation with UploadFile in a single request per
	#https://github.com/tiangolo/fastapi/issues/657

	model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	results = model( Image.open(BytesIO(await file.read())))
	json_results = results_to_json(results,model)
	return json_results

@app.get("/about/")
def about_us(request: Request):
	'''
	Display about us page
	'''

	return templates.TemplateResponse('about.html', 
			{"request": request})

if __name__ == '__main__':
    import uvicorn
    import os.path
    
    #make the app string equal to whatever the name of this file is
    app_str = os.path.splitext(os.path.basename(__file__))[0] + ':app'
    
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
