# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:53:16 2019

@author: paprasad
"""
import cv2
import numpy as np

kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_erode = np.ones((1, 1), np.uint8)

def whiteBack(roi):
    avg = np.mean(roi)
    
    if(avg<100):
        return True
    else:
        return False

def sharp_img(img,kernel):
    img = cv2.filter2D(img, -1, kernel)
    return img

def dialte_img(img,kernel):
    img = cv2.dilate(img, kernel, iterations = 1)
    return img

img = cv2.imread('./Traning Images/car10.jpg')

org_img = img
cv2.imshow('original_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = sharp_img(img,kernel_sharp)
#cv2.imshow('sharp_img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('cvt_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


ret, img = cv2.threshold(img, 65, 255, 0)
cv2.imshow('thershold_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = 0
for index,contour in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(contour)
    ratio = float(w) / float(h)
    roi = img[y:y+h,x:x+w]
    if ratio < 1 and whiteBack(roi):
        cv2.rectangle(org_img, (x,y), (x+w,y+h), (0,255,255), 1)  
        cnt = cnt+1
        print(x,y,w,h)
        
print(cnt)        
cv2.imshow('countour Img',org_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




#
#img = dialte_img(img, kernel_erode)
#cv2.imshow('Dilate Img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#classifier_xml = "russ.xml"
#numberplate = cv2.CascadeClassifier(classifier_xml)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plates = numberplate.detectMultiScale(gray,1.3,5)
#
##def find_plate_number(img,classifier_xml = "russ.xml"):
##    numberplate = cv2.CascadeClassifier(classifier_xml)
##    #numberplate = cv2.threshold(numberplate, 127, 255, 0)
##    frame = cv2.imread(img)
##    imgGrayscale, imgThresh = Preprocess.preprocess(frame)
##
##    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
##    gray = imgGrayscale
##    
##    plates = numberplate.detectMultiScale(gray,1.3,5)
##    print(plates)
##    numbers = []
##    for plate in plates:
##        x,y,w,h = plate
##        roi = frame[y:y+h,x:x+w]
##        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
##        numbers.append(read_number(roi))
##    return numbers,frame,roi   
#


