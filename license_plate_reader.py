# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:48:25 2019

@author: Pawan
"""

import numpy as np
import cv2
import sys
import os
import Plate
import Chars
import pytesseract

import ImagePreprocessing
from pytesseract import image_to_string

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"



def number_processing(number):
    number = number.replace(' ','')
    #print(number)
    return number

def tessercat_number(licPlate):
#    x = licPlate.intBoundingRectX
#    y = licPlate.intBoundingRectY
#    h = licPlate.intBoundingRectHeight
#    w = licPlate.intBoundingRectWidth
#    imgGrayscale, imgThresh = ImagePreprocessing.preprocess(car_image) 
#    roi = imgThresh[y:y+h,x:x+w]
#    roi_resized = cv2.resize(roi, (20, 30)) 
#    cv2.imshow('licPlate.imgThresh',licPlate.imgGrayscale)
#    cv2.waitKey(0)
    #plate_resized = licPlate.imgThresh
    plate_resized = cv2.resize(licPlate.imgGrayscale, (180, 50))   
#    cv2.imshow('plate_resized',plate_resized)
#    cv2.waitKey(0)
    plate_resized = ImagePreprocessing.maximizeContrast(plate_resized)
#    cv2.imshow('maximizeContrast',plate_resized)
#    cv2.waitKey(0)
    plate_resized = cv2.GaussianBlur(plate_resized, (3,3), 0)
#    cv2.imshow('Burred',plate_resized)
#    cv2.waitKey(0)
    ret,imgThresh  = cv2.threshold(plate_resized, 120, 255, cv2.THRESH_BINARY)
#    cv2.imshow('imgThresh',imgThresh)
#    cv2.waitKey(0)
    text = pytesseract.image_to_string(plate_resized, lang='eng')  
    #print("i am here",text)
    return  number_processing(text) 

def license_plate_reader(car_image):
    car_image = cv2.imread(car_image)
    cv2.imshow('car_image',car_image)
    cv2.waitKey(0)
    if car_image is None:
        print("Not able to read image please check image name of image path")
        return 0
    else:
        KNNobject = Chars.LoadAndTrainKNN()
        if KNNobject is None:
            print("\n Problem in loading the KNN object")
        else:
            All_possible_plates = Plate.find_plates(car_image)
            All_possible_plates = Chars.detectCharsInPlates(All_possible_plates)
            
        if len(All_possible_plates) == 0:                          
            print("\nNo plate detected please adjust the camera angle\n")  
        else:                                                       
            All_possible_plates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
            licPlate = All_possible_plates[0]
            tessercat_OCR = tessercat_number(licPlate)
            if tessercat_OCR is None:
                print("\n no number detected by Tesseract OCR")
            if len(licPlate.strChars) == 0:                     
                print("\nno characters were detected from KNN\n\n")  
                return                                   
            
            return licPlate.strChars,tessercat_OCR
    
    
    
    

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("invalid number argument \n correct usage is : \n license_plate_reader  <image_name>" )
        
    else:
        print(sys.argv[1])
        car_image = sys.argv[1]        
        ret = license_plate_reader(car_image)
        print(ret)
    