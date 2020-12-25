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


class face_detector:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def preprocess_input(self, img):
        # read image
        
        ht = img.shape[0]
        wd = img.shape[1]
        cc = img.shape[2]
        # print(f"face size: {ht,wd}")

        #   print(img.shape)

        # create new image of desired size and color (blue) for padding
        color = (0,0,0)
        hh = 160
        ww = 160
        diff_h = hh - ht 
        diff_w = ww - ht

        if diff_h > 0 and diff_w > 0:
            result = np.full((hh,ww,cc), color, dtype=np.uint8)
            xx = (ww - wd) // 2
            yy = (hh - ht) // 2
            result[yy: yy + ht, xx: xx + wd] = img
           
        else:
            result = cv2.resize(img, (hh, ww), interpolation=cv2.INTER_AREA)
            
            
        return result
        
    def draw_bbox(self, cv_img, state="Unknown"):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.4
        fontColor              = (255,255,255)
        lineType               = 2
    
        
        start_time = time.time()
        gray =  cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        Image = cv_img
        faces = self.detector(gray)
        fps = (1.0 / (time.time() - start_time))
        fps = "FPS:"+ str(fps)
        face_pos_list =[]
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right()-face.left()+10
            h = face.bottom()-face.top()+10
            cv2.rectangle(Image, (x, y), (x + w, y + h), (255, 25, 10), 1)
            face_crop = Image[y - 20 : y + h, x: x + w + 4]
            preprocess_input = self.preprocess_input(face_crop)
            crop_face = np.expand_dims(preprocess_input, 0)
            face_pos_list.append([crop_face,[x,y,w,h]])
            cv2.line(Image, (x, y), (x + int(w / 5), y), (255,25,10),3)
            cv2.line(Image, (x + int((w/ 5)* 4), y), (x + w, y), (255,25,10),3)
            cv2.line(Image, (x, y), (x, y + int(h / 5)), (255,25,10), 3)
            cv2.line(Image, (x +w, y), (x + w, y + int(h / 5)),(255,25,10), 3)
            cv2.line(Image, (x, (y + int(h /5* 4))), (x, y+h),(255,25,10), 3)
            cv2.line(Image, (x, (y+h)), (x + int(w / 5), y + h),(255,25,10), 3)
            cv2.line(Image, (x + int((w/ 5)* 4), y+ h), (x +w, y + h), (255,25,10), 3)
            cv2.line(Image, (x + w, (y + int(h / 5 * 4))), (x + w, y + h), (255, 25, 10), 3)

            # cv2.putText(Image, "NAME : {}".format("ui"), (x + w + 20, y + 100), font, fontScale, fontColor, lineType)
            # cv2.putText(Image, "Pos : X:{} Y:{}".format(x, y), (x + w + 20, y + 120), font, fontScale, fontColor, lineType)
        cv2.putText(Image, fps, (10, 50), font, fontScale, fontColor, lineType)
        
        yield Image, face_pos_list