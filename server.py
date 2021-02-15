from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import torch
import base64
import random

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

#so we can read main.css
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

#for bbox plotting
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]

@app.get("/")
def home(request: Request):
	'''
	Returns html jinja2 template render for home page form
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

def plot_one_box(xyxy, img, color = (255,255,255), label=None, line_thickness=None):
	#function based on yolov5/utils/plots.py plot_one_box()
	#implemented using PIL instead of OpenCV to avoid converting between PIL and OpenCV formats

	color = color or [random.randint(0, 255) for _ in range(3)]
	width, height = img.size

	xyxy = [int(width * xyxy[0]),
			int(height * xyxy[1]),
			int(width * xyxy[2]),
			int(height * xyxy[3]),
		]
	
	draw = ImageDraw.Draw(img)

	draw.rectangle(xyxy, outline=color, width = 3)

	if label:
		#drawing text in PIL is much harder than OpenCV due to needing ImageFont class
		#for some reason PIL doesn't have a default font that scales...
		try:
			#works on Windows
			fnt = ImageFont.truetype("arial.ttf", 36)
		except:
			'''
			linux might have issues with the above font, so adding this section to handle it
			this method is untested. based on:
			https://stackoverflow.com/questions/24085996/how-i-can-load-a-font-file-with-pil-imagefont-truetype-without-specifying-the-ab
			'''
			fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 36, encoding="unic")

		txt_width, txt_height = fnt.getsize(label)

		draw.rectangle([xyxy[0],xyxy[1]-txt_height-2, xyxy[0]+txt_width+2, xyxy[1]], fill = color)
		draw.text((xyxy[0],xyxy[1]-txt_height), label, fill=(255,255,255), font = fnt)

@app.post("/")
async def detect_via_web_form(request: Request,
							file: UploadFile = File(...), 
							model_name: str = Form(...)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). Intended for human (non-api) users.
	Returns: HTML template render showing bbox data and base64 encoded image
	'''

	#NOTE: Can't use pydantic validation with UploadFile in a single request per
	#https://github.com/tiangolo/fastapi/issues/657

	#this can be preloaded to not have to run each time
	model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	img = Image.open(BytesIO(await file.read()))

	results = model(img) #get YOLO results on the input image
	json_results = results_to_json(results,model)

	#plot bboxes on the image
	for bbox_list in json_results:
		for bbox in bbox_list:
			label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
			plot_one_box(bbox['normalized_box'], img, label=label, color=colors[int(bbox['class'])], line_thickness=3)

	#base64 encode the image string so we can render it in HTML
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

	encoded_json_results=str(json_results).replace("'",r"\'").replace('"',r'\"')

	print(encoded_json_results)
	return templates.TemplateResponse('show_results.html', {
			'request': request,
			'image_base64': img_str,
			'bbox_data': json_results,
			'bbox_data_str': encoded_json_results,
		})

@app.post("/detect/")
async def detect_via_api(request: Request,
						file: UploadFile = File(...), 
						model_name: str = Form(...)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). Intended for API usage.
	Returns: JSON results of running YOLOv5 on the uploaded image
	'''

	#this can be preloaded to not have to run each time
	model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	results = model(Image.open(BytesIO(await file.read()))) #get YOLO results on the input image
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
