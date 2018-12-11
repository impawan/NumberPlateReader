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

cv2.imshow("imgray",imgray)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#cv2.imshow("thresh",thresh)

def get_number_plate(plate_area_1):
    ret, plate_area = cv2.threshold(plate_area, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(plate_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for index,contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        roi = plate_area[y:y+h,x:x+w]
        cv2.re
        print(whiteBack(roi),index)
        cv2.rectangle(plate_area, (x,y), (x+w,y+h), (0,255,0), 1)
        if whiteBack(roi):
            return roi 
        else:
            return None 
        
    

    
    

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_center(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def number_plate(contour):
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), rect_angle = rect
    print(x,y,width,height,rect_angle)
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    margin = 10
    cv2.drawContours(img,[box],0,(0,0,255),2)
    
    (o_x,o_y,o_w,o_h) = cv2.boundingRect(contour)
    o_x=o_x-margin
    o_y=o_y-margin
    o_h=o_h+margin*2
    o_w=o_w+margin*2
    roi = img[o_y:o_y+o_h,o_x:o_x+o_w]
    print(whiteBack(roi))
    rotated_im = rotate_center(roi,rect_angle)
    #roi = get_number_plate(rotated_im)
    return roi 

def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area

	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True
    




def whiteBack(roi):
	avg = np.mean(roi)
	if(avg>=120):
		return True
	else:
 		return False

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, approx, -1, (0,255,0), 5)
#cv2.drawContours(im,contours[156],-1,(255,0,0),1)
#cv2.imshow("conto", im)
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
    #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
    #cv2.putText(img,str(index),(x,y),font,0.5,(0,255,255),1,cv2.LINE_4)
    #if ratio >= 3.9 and ratio <= 4.1:
    area = cv2.contourArea(contour)
    #print(area)
    #print(ratioCheck(area,w,h),whiteBack(roi),index)
    if ratio >= 1.9 and ratio <= 2.2 and whiteBack(roi):# and area >=22200 and area <=22000 :
        print(whiteBack(roi))
        print("yes")
        print(index,ratio)
        #roi = number_plate(thresh)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(img,str(index),(x,y),font,0.5,(0,255,255),1,cv2.LINE_4)
#    
#cv2.imshow("number_plate",roi)        
cv2.imshow("Keypoints", img)    
cv2.waitKey(0)
cv2.destroyAllWindows()