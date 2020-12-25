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
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from face_recognition_system import face_detector
from Service.FaceRecognitionCore import  FaceRecognition
from Utils.thaitext import drawText



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.face_detector = face_detector()
        self.FaceRecognition = FaceRecognition()


    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()

            if ret:
               try:
                    x =  self.face_detector.draw_bbox(cv_img)
                    for cv_img, faces in x:
                        # print(len(faces))
                        for i, face in enumerate(faces, 1):
                            # print(face[1])
                            encode_face_data = self.FaceRecognition.encode_face_data(face[0])
                            result_idx = self.FaceRecognition.match(encode_face_data, 1)
                            result = self.FaceRecognition.getface(result_idx,theshold=7.5)
                            name = result['name']
                            student_id = result['student_id']
                            nickname = result['nickname']
                            classroom = result['class']
                            image_path = result['image_path']
                            added_time = result['added_time']
                            print(f"face: {i} name: {name}, student_id: {student_id} nickname: {nickname}")
                            cv_img = drawText(cv_img, f"face_id :{i} {name} {nickname}", pos=(face[1][0], face[1][1]-70, face[1][2], face[1][3]), fontSize=18, color=(255, 255, 255))
                            
                    
               except:
                    pass
               self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWS_FACE_RECOGNITION")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())