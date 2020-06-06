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



face_detector  = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
landmark_predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
COLOR = [255, 179 , 0]

    
    

def check(temp, status, sets=37.5):
    color_check = (0,255,0)
    status_color = (255,255,255)
    if status == 'Teacher':
        status_color = (255,179,0)
    elif status == 'Student':
        status_color = (0,25,179)
    elif status == 'Grad':
        status_color = (0,255,179)
    elif status == 'Mad':
        status_color = (0,50,179)

    elif status == 'Unknown':
        status_color = (0,0,255)

    if temp>=sets:
        color_check = (0,50,255)
        is_suspect = True
        
        return  color_check,is_suspect,status_color
        

  
    elif temp<sets:
        color = (0,255,0)
        is_suspect = False
        return  color_check,is_suspect,status_color
        
 
def draw_detect(Image, x, y, w, h,temp,color_check,is_suspect,status,status_color,cid,name,color,thick=3):
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.rectangle(Image,(x,y),(x+w,y+h),status_color,1)
    cv2.line(Image, (x, y), (x + int(w / 5), y), status_color, thick)
    cv2.line(Image, (x + int((w/ 5)* 4), y), (x + w, y), status_color, thick)
    cv2.line(Image, (x, y), (x, y + int(h / 5)), status_color, 2)
    cv2.line(Image, (x +w, y), (x + w, y + int(h / 5)), status_color, thick)
    cv2.line(Image, (x, (y + int(h /5* 4))), (x, y+h), status_color, thick)
    cv2.line(Image, (x, (y+h)), (x + int(w / 5), y + h), status_color, thick)
    cv2.line(Image, (x + int((w/ 5)* 4), y+ h), (x +w, y + h), status_color, thick)
    cv2.line(Image, (x+ w, (y + int(h /5*4))), (x+ w, y + h), status_color, thick)
    cv2.putText(frame,"Status: {}".format(status), (x+w+20,y+40), font, 1,status_color,lineType)
    cv2.putText(frame,"CID : {}".format(cid), (x+w+20,y+80), font, fontScale,fontColor,lineType)
    cv2.putText(frame,"NAME : {}".format(name), (x+w+20,y+100), font, fontScale,fontColor,lineType)
    cv2.putText(frame,"Themerature : {}".format(temp), (x+w+20,y+120), font, fontScale,color_check,lineType)
    cv2.putText(frame,"Is suspect : {}".format(is_suspect), (x+w+20,y+140), font, fontScale,color_check,lineType)
      


    
cap = cv2.VideoCapture(0)
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2
while True:
    _, frame = cap.read()
    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
   
    
  
    faces_frame = face_detector.detectMultiScale(gray, 1.3, 5)

    temp = 36.5
    status_list = [ "Student","Teacher","Mad","Unknown","Grad"]
    name_list = [ "Pattanun Numpong","Samorn Numpong","Nuntapat Numpong","Unknown"]
    name = choice(name_list)
    status = choice(status_list)
    cid = randint(000000,999999)
    
    print("Name:",name, "Stutus:",status, "CID", cid )
    for (x,y,w,h) in faces_frame:
        face = frame[y-20:y+h,x:x+w+40]
   
        color_check = check(temp,status)[0]
        is_suspect = check(temp,status)[1]
        status_color = check(temp,status)[2]
        
        draw_detect(frame,x,y,w,h,temp,color_check,is_suspect,status,status_color,cid, name,(0,255,179),2)
        
        

    faces = detector(gray)
    face_pos_list =[]
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        face_pos = "Position: "+" X1: "+str(x1)+" Y1: "+str(y1)+" X2: "+str(x2)+" Y2: "+str(y2)
        face_pos_list.append(face_pos)
        for i, face_pos in enumerate(face_pos_list,1):
            cv2.putText(frame,"ID: {} ".format(i)+face_pos, (300,70*i),font, fontScale,fontColor,lineType)
            

        landmarks = landmark_predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # print(x,y)
            cv2.circle(frame, (x, y), 2, COLOR, -1)   
            
    fps = (1.0 / (time.time() - start_time))
    fps = "FPS:"+ str(fps)
    print("FPS",fps)
    cv2.putText(frame,fps, (10,50), font, fontScale,fontColor,lineType)
    cv2.imshow('FaceRecognition',frame)
    
    
    if (cv2.waitKey(1)& 0xFF) == ord("q"):
        cv2.destroyAllWindows()
        break
    


