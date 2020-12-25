from Service.FaceRecognitionCore import FaceRecognition
import os
import  pendulum
import json
from tqdm import  tqdm

if __name__ == "__main__":
    path = './src/Database/6-3/'
    FaceRecognition = FaceRecognition()
    FaceRecognition.add_face(path)
  


            