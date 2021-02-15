''' Example client sending POST request to server and printing the YOLO results
'''

import requests as r
import json
from pprint import pprint

def send_request(image = '../images/zidane.jpg', model_name = 'yolov5s'):
	
	files = {'file': open(image , "rb")} #pass the files here

	#pass the other form data here
	other_form_data = {'model_name': model_name}

	res = r.post("http://localhost:8000", 
					data=other_form_data, 
					files = files)

	pprint(json.loads(res.text))

if __name__ == '__main__':
	send_request()