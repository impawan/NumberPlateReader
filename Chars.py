# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:53:55 2019

@author: Pawan
"""

# DetectChars.py
import os

import cv2
import numpy as np
import math
import random


import ImagePreprocessing
import PossibleChar



kNearest = cv2.ml.KNearest_create()

#Static values to to find the contour having charcacters

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80


MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0


MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100






kNearest = cv2.ml.KNearest_create()



'''
The method KNNobjectLoadTrain is used to load the KNN object 
'''
def LoadAndTrainKNN():
    #allContoursWithData = []                
    #validContoursWithData = []              

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  
    except:                                                                                 
        print("error, unable to open classifications.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        


    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 
    except:                                                                                 
        print("error, unable to open flattened_images.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       

    kNearest.setDefaultK(1)                                                             

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           

    return True  



def detectCharsInPlates(listOfPossiblePlates):


    if len(listOfPossiblePlates) == 0:          
        return listOfPossiblePlates             


    for possiblePlate in listOfPossiblePlates:          

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = ImagePreprocessing.preprocess(possiblePlate.imgPlate)     
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)


        if (len(listOfListsOfMatchingCharsInPlate) == 0):			
            possiblePlate.strChars = ""
            continue						
    

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
        

       
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
    return listOfPossiblePlates


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                       
    contours = []
    imgThreshCopy = imgThresh.copy()

            
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              
            listOfPossibleChars.append(possibleChar)       
       

    return listOfPossibleChars



def checkIfPossibleChar(possibleChar):
            
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False



def findListOfListsOfMatchingChars(listOfPossibleChars):
            
    listOfListsOfMatchingChars = []                  

    for possibleChar in listOfPossibleChars:                        
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        

        listOfMatchingChars.append(possibleChar)                

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                            
                                                
      

                                                
        listOfListsOfMatchingChars.append(listOfMatchingChars)      

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                
                                                
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             
       

        break       

   

    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
            
    listOfMatchingChars = []                

    for possibleMatchingChar in listOfChars:                
        if possibleMatchingChar == possibleChar:    
                                                    
            continue                                
        
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        

    return listOfMatchingChars                 



def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))



def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                          
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
    else:
        fltAngleInRad = 1.5708                         
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       

    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         
                        
                    else:                                                                       
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           
    

    return listOfMatchingCharsWithInnerCharRemoved


def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ''
    #tesseract_text = []
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     

    for currentChar in listOfMatchingChars:                                         
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
       
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        
        npaROIResized = np.float32(npaROIResized)  
        #text = pytesseract.image_to_string(npaROIResized, lang='eng')             
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              
        strCurrentChar = str(chr(int(npaResults[0][0])))            
        strChars = strChars + strCurrentChar                        



    return strChars








