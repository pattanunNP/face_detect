import numpy as np
import dlib, cv2, os
from PIL import  Image

class Prepare:

    def __init__(self):
        self.facedetector = dlib.get_frontal_face_detector()
        print("Ready")

 

    def preprocess_input(self, data):
        print(os.path.isfile(data))
        image = Image.open(data)
        image = np.asarray(image)
        print(image.shape)

        gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.facedetector(gray)
        face_pos_list =[]
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right()
            h = face.bottom()
            crop_face = image[y + 100 : h + 100, x + 100 : w + 100]
        crop_face = cv2.resize(crop_face,(160,160))
        crop_face = np.expand_dims(crop_face, 0)
        print(crop_face.shape)
        return crop_face