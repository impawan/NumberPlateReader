
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string


        
def read_number(roi):
    plate_im = Image.fromarray(roi)
    text = pytesseract.image_to_string(plate_im, lang='eng')
    print (text)
    return text
    
def find_plate_number(img,classifier_xml = "russ.xml"):
    numberplate = cv2.CascadeClassifier(classifier_xml)
    #numberplate = cv2.threshold(numberplate, 127, 255, 0)
    frame = cv2.imread(img)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    plates = numberplate.detectMultiScale(gray,1.3,5)
    numbers = []
    for plate in plates:
        x,y,w,h = plate
        roi = gray[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        numbers.append(read_number(roi))
    return numbers,frame,roi   



img = "test.jpeg"
text,frame,roi  = find_plate_number(img)

cv2.imshow('Face Detection',frame)

cv2.imshow('roi',roi)    
cv2.waitKey(0)
cv2.destroyAllWindows()   
