# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:29:20 2019

@author: Pawan
"""



import cv2
import numpy as np
import math


class PossibleChar:
    ''' This method gets the contour through consrtuctor 
    find the bounding rectangle calculates the rectangle area find the x,y center find the daigonal size calculate the aspect are '''
    def __init__(self, _contour):
        self.contour = _contour
  
        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)








