# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:03:33 2020

@author: Haseeb
"""

from PIL import Image
import cv2
import pytesseract
import numpy as np
#uncomment below line if you dont want to set "Environment variables" path
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from  PythonScripts  import globals

def meterno_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    th3 = cv2.dilate(th3, kernel, iterations=1)
    th3=255-th3
    text = pytesseract.image_to_string(th3)
    start_reading=False
    meter_reading=''
    for i in range(len(text)):
            if text[i].isnumeric():
                meter_reading=meter_reading+text[i]
        

    return meter_reading[-7:]