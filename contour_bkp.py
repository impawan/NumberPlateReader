# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:10:53 2018

@author: Pawan
"""

import numpy as np
import cv2 
img = cv2.imread('test3.jpg')
#im = np.array(im, dtype=np.uint8)

im = cv2.GaussianBlur(img, (5,5), 0)

cv2.imshow("blur",im)

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(imgray, 127, 255, 0)
cv2.imshow("thresh",thresh)


def whiteBack(roi):
	avg = np.mean(roi)
	if(avg>=130):
		return True
	else:
 		return False

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, contours[44], -1, (0,255,0), 5)
#cv2.drawContours(im,contours,-1,(255,255,0),1)
#cv2.imshow("Keypoints", im)
#
font = cv2.FONT_HERSHEY_SIMPLEX
#cnt = contours[0]
#
#(x,y,w,h) = cv2.boundingRect(cnt)
#cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)
for index,contour in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(contour)
    #print(x,y,w,h)
    ratio = w/h
    #print(ratio)
    roi = im[y:y+h,x:x+w]
    
    #if ratio >= 3.9 and ratio <= 4.1:
    if ratio >= 4.5 and ratio <= 5 and whiteBack(roi) :
        print(whiteBack(roi))
        print("yes")
        print(index,ratio)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(img,str(index),(x,y),font,0.5,(0,255,255),1,cv2.LINE_4)
#    

cv2.imshow("Keypoints", img)    
cv2.waitKey(0)
cv2.destroyAllWindows()
