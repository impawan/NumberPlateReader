# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:10:53 2018

@author: Pawan
"""

import numpy as np
import cv2 
img = cv2.imread('test3.jpg')
import pytesseract
#im = np.array(im, dtype=np.uint8)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
im = cv2.GaussianBlur(img, (5,5), 0)

cv2.imshow("blur",im)

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
max_width = 500
max_height = 250

ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#cv2.imshow("thresh",thresh)



def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area

	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True



def image_processing(img):
    width = int(img.shape[1] * max_width / 100)
    height = int(img.shape[0] * max_height / 100)
    dim = (width, height)
    img=cv2.resize(img, (100,50),dim,interpolation=cv2.INTER_CUBIC)
    retval, thresh_gray = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    return img

def find_plate(rotated_im):
    text = pytesseract.image_to_string(rotated_im, lang='eng')
    if text is None:
        return False
    else:
        return text

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


def whiteBack(roi):
	avg = np.mean(roi)
	if(avg>=120):
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
#    rect = cv2.minAreaRect(contour)
#    (x, y), (width, height), rect_angle = rect
#    #print(x,y,width,height,rect_angle)
#    
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
    margin = 10
#    cv2.drawContours(img,[box],0,(0,0,255),2)
#    print(index)
    (o_x,o_y,o_w,o_h) = cv2.boundingRect(contour)
    if whiteBack(thresh[o_y:o_y+o_h,o_x:o_x+o_w]):
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), rect_angle = rect
        o_x=o_x-margin
        o_y=o_y-margin
        print(index)
        o_h=o_h+margin*2
        o_w=o_w+margin*2
        roi = img[o_y:o_y+o_h,o_x:o_x+o_w]
        rotated_im = rotate_center(roi,rect_angle)
        number = find_plate(rotated_im)
        if number is not None:
            print(number)
   

cv2.imshow("Keypoints", img)    
cv2.waitKey(0)
cv2.destroyAllWindows()