
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string
import Preprocess
import PossibleChar

'''

This is the exmaple of multiline comment 

'''
        
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80
 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"






def find_contours(plate):
    im2, contours, hierarchy = cv2.findContours(plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(im2, (x,y), (x+w,y+h), (255,0,0), 1)
        
    cv2.imshow('im_contours',im2)     
    cv2.waitKey(0)
    return 0 



def read_number(roi):
    #plate_im = Image.fromarray(roi)
    plate_im = roi
    plate_im = cv2.resize(plate_im, (0, 0), fx = 1.6, fy = 1.6)
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(roi) 

    text = pytesseract.image_to_string(plate_im, lang='eng')
    return text
    
def find_plate_number(img,classifier_xml = "russ.xml"):
    numberplate = cv2.CascadeClassifier(classifier_xml)
    #numberplate = cv2.threshold(numberplate, 127, 255, 0)
    frame = cv2.imread(img)
    imgGrayscale, imgThresh = Preprocess.preprocess(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = imgGrayscale
    
    plates = numberplate.detectMultiScale(gray,1.3,5)
    print(plates)
    numbers = []
    for plate in plates:
        x,y,w,h = plate
        roi = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        numbers.append(read_number(roi))
    return numbers,frame,roi   



