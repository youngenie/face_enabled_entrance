#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:57:31 2019

@author: youngjinlee

@title: face_recognition database(manual)
"""

import numpy as np
import cv2 as cv
import PIL
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

import face_recognition
import argparse
import pickle
import shutil
import pathlib

"""
1. Load the pre-trained model, FaceNet (BOTTLENECK)
"""
# Define FRmodel
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
# Loading pre-trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel) # bottleneck(critical)
print("Total Params:", FRmodel.count_params())
os.chdir('/Users/youngjinlee/Desktop/bdt_project')

"""
2. Load the current database (if you are creating a new database, remove '#' in the next line)
"""
#db = {}
db = np.load('database.npy')
db=db[()]
db   

"""
3. The following iterates over 'num' files in the directory that corresponds with 'new_name'. \\
The image file names should be integers in ascending order.
you also need to set 'start' for selecting images you want to start from.
"""
# Define the name for the person you want to add to the database.
new_name = 'bdt_malee_950523'
num = 1
start = 30

for i in range(num):
    i += start
    file_name = str(i) + '.jpg'
    print(file_name)
    
    # Read the image file from the following directory 
    image = cv.imread("/Users/youngjinlee/Desktop/bdt_project_fr/images_database/" + new_name +'/'+ file_name, 1)
    
    # convert the image from BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb, model="cnn") # bottleneck
    
    # top, right, bottom, left
    # x : boxes[0][3]
    # y : boxes[0][0]
    # x + w : boxes[0][1]
    # y + h : boxes[0][2]
    
    # Find the largest anchor box in case there are multiple faces detected
    largest = 0
    for i in range(len(boxes)):
        current = boxes[i][2] - boxes[i][0]
        print(current)
        if current > largest:
            largest = current
            j = i
    box = boxes[j]
    
    # Based on the (x,y,w,h) coordinates for the box, crop the image
    crop_img = image[box[0]:box[2], box[3]:box[1]]
    
    # Resize the image to 96 x 96
    dst = cv.resize(crop_img, dsize=(96,96), interpolation=cv.INTER_AREA)
    
    # Save the cropped image in the following directory
    im = Image.fromarray(dst)
    im.save("cropped_images_database/" + new_name + "/" + file_name)
    
    # encode the image and save the vector to the database with the corresponding key value
    key = new_name
    db.setdefault(key, [])    
    db[key].append(img_to_encoding(dst, FRmodel))
    
    # save the database
    np.save('database.npy', db)
    