from PIL import Image
import base64
import numpy as np
import requests
import json
import cv2 
import dlib
import sys
from random import randint
from random import uniform
from random import choice
import time

file = './rick-test.jpg'

def predict(file):
    url = 'http://0.0.0.0:5000/face_analysis_api/predict?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiTWFzdGVyS2V5IiwiUGFzc3dvcmQiOiJMb2Rhc2hBUEkyMDIwIn0.JJP6QzScKE-qPInyJIzaIa2_hCk-Y-93rdmbuTz1N9o'
    # url = 'https://botnoi-facerecognition-api.herokuapp.com/'
    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    encoded_string= str(encoded_string)
    encoded_string =  encoded_string[2:-1]
    # print(encoded_string)
    start_time = time.time()
    # print("Round:",i+1)
    r = requests.post(url,json={"image":encoded_string})
   
    time_res = time.time() - start_time
    print("Time:",time_res)
predict(file)