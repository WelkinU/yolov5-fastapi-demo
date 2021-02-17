''' Example client sending POST request to server (localhost:8000/detect/)and printing the YOLO results
'''

import requests as r
import json
from pprint import pprint

def send_request(file_list = ['./images/zidane.jpg'], model_name = 'yolov5s'):

	#upload multiple files as list of tuples
	files = [('file_list', open(file,"rb")) for file in file_list]

	#pass the other form data here
	other_form_data = {'model_name': model_name}

	res = r.post("http://localhost:8000/detect/", 
					data= other_form_data, 
					files = files)

	pprint(json.loads(res.text))

if __name__ == '__main__':
	send_request(['./images/bus.jpg',
				'./images/zidane.jpg',
				])