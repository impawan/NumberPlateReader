# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:46:51 2019

@author: Pawan
"""



''' Detecting the plate '''
import os
import cv2
import numpy as np
import math
import random
import ImagePreprocessing



import PossibleChar
import Chars
import PossiblePlate

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5




def extractPlate(car_image, listOfMatchingChars):
    '''
    
    '''
    possiblePlate = PossiblePlate.PossiblePlate()          
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        
   
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

           
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = Chars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            

            
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = car_image.shape      

    imgRotated = cv2.warpAffine(car_image, rotationMatrix, (width, height))       

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    possiblePlate.imgPlate = imgCropped         

    return possiblePlate


def findPossibleCharsInImage(imgThresh):
    '''
        Read the Thresh image
        Find all the contours 
        create object of all possible contours
        check if there are any characters in bounding rectangle formed by contours 
        if yes, append that instance in the list
        return the  list 
    '''
    listOfPossibleChars = []               
    
    imgThreshCopy = imgThresh.copy()
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    height, width = imgThresh.shape

    for i in range(0, len(contours)):                        
        possibleChar = PossibleChar.PossibleChar(contours[i])
        if Chars.checkIfPossibleChar(possibleChar):                  
            listOfPossibleChars.append(possibleChar)                        

    return listOfPossibleChars

def find_plates(car_image):
    '''
    This method find the all possible plate for image in argument
    find the all the contour that have character with in their bounding rectangle
    '''
    all_plates = []
    height, width, channel = car_image.shape
    imgGrayscale, imgThresh = ImagePreprocessing.preprocess(car_image)  
    
    listOfPossibleCharsInImage = findPossibleCharsInImage(imgThresh)
    
    listOfListsOfMatchingCharsInScene = Chars.findListOfListsOfMatchingChars(listOfPossibleCharsInImage)
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   
        possiblePlate = extractPlate(car_image, listOfMatchingChars)         

        if possiblePlate.imgPlate is not None:                         
            all_plates.append(possiblePlate)                  

    #print("\n" + str(len(all_plates)) + " possible plates found") #need to change this 


    return all_plates