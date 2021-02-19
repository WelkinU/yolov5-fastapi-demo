from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

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

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x']
model_dict = {model_name: None for model_name in model_selection_options}

@app.get("/")
def home(request: Request):
	'''
	Returns html jinja2 template render for home page form
	'''

	return templates.TemplateResponse('home.html', {
			"request": request,
			"model_selection_options": model_selection_options,
		})


def results_to_json(results, model):
	''' Converts yolo model output to json (list of list of dicts)
	'''
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
							file_list: List[UploadFile] = File(...), 
							model_name: str = Form(...),
							img_size: int = Form(...)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). Intended for human (non-api) users.
	Returns: HTML template render showing bbox data and base64 encoded image
	'''

	#Not sure how to use pydantic validation with UploadFile in a single request. 
	#See: https://github.com/tiangolo/fastapi/issues/657
	if model_name not in model_selection_options:
		return {"status": "error", "message": "Invalid YOLO model name"}

	if model_dict[model_name] is None:
		model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	img_batch = [Image.open(BytesIO(await file.read())) for file in file_list]

	results = model_dict[model_name](img_batch.copy(), size = img_size) #get YOLO results on the input image
	json_results = results_to_json(results,model_dict[model_name])

	img_str_list = []
	#plot bboxes on the image
	for img, bbox_list in zip(img_batch, json_results):
		for bbox in bbox_list:
			label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
			plot_one_box(bbox['normalized_box'], img, label=label, 
					color=colors[int(bbox['class'])], line_thickness=3)

		#base64 encode the image so we can render it in HTML
		buffered = BytesIO()
		img.save(buffered, format="JPEG")
		img_str_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

	#escape the apostrophes in the json string representation
	encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

	return templates.TemplateResponse('show_results.html', {
			'request': request,
			'bbox_image_data_zipped': zip(img_str_list,json_results), #zip here, instead of in jinja2 template
			'bbox_data_str': encoded_json_results,
		})


@app.post("/detect/")
async def detect_via_api(request: Request,
						file_list: List[UploadFile] = File(...), 
						model_name: str = Form(...),
						img_size: Optional[int] = Form(640)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). Intended for API usage.
	Returns: JSON results of running YOLOv5 on the uploaded image
	'''

	if model_name not in model_selection_options:
		return {"status": "error", "message": "Invalid YOLO model name"}

	if model_dict[model_name] is None:
		model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	img_batch = [Image.open(BytesIO(await file.read())) for file in file_list]
	
	results = model_dict[model_name](img_batch, size = img_size)
	json_results = results_to_json(results,model_dict[model_name])

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
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--precache-models', action='store_true', help='pre-cache all models in memory upon initialization')
	opt = parser.parse_args()

	if opt.precache_models:
		#pre-load models
		model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True) 
						for model_name in model_selection_options}
	
	#make the app string equal to whatever the name of this file is
	app_str = 'server:app'
	uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
