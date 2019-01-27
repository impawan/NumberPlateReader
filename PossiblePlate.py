# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:54:55 2019

@author: Pawan
"""

import cv2
import numpy as np


class PossiblePlate:

   
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""





