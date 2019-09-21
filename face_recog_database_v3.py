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

# Define FRmodel
FRmodel = faceRecoModel(input_shape=(3, 96, 96))# bottle neck
# Loading pre-trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel) # bottleneck(critical)
print("Total Params:", FRmodel.count_params())

# load the input image and convert it from BGR to RGB

os.chdir('/Users/youngjinlee/Desktop/face_recognition_low')

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True, help="name of the person")
args = vars(ap.parse_args())
print('args', args)
# load the known faces and embeddings
print("[INFO] loading encodings...")
    
# Access 'pics' folder on Raspberry pi and extract the filename at the last index.
lst = []
for item in os.listdir('/Volumes/sh/pics'):
    if not item.startswith('.') and os.path.isfile(os.path.join('/Volumes/sh/pics', item)):
        comp = int(item.split('.')[0])
        lst.append(comp)    
lst = sorted(lst)

filename = str(lst[len(lst)-1]) + ".jpg"
filename

# Define the name for this new person[arg_parse]
new_name = args["name"]
        
# Read the image file
image = cv.imread("/Volumes/sh/pics/" + filename, 1)
print(image)        

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
        
boxes = face_recognition.face_locations(rgb, model="cnn") # bottleneck
        
# x : boxes[0][3]
# y : boxes[0][0]
# x + w : boxes[0][1]
# y + h : boxes[0][2]
        
# Crop the image
crop_img = image[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][1]]
# Make the size of the image 96 x 96
dst = cv.resize(crop_img, dsize=(96,96), interpolation=cv.INTER_AREA)


# cropped image 저장
im = Image.fromarray(dst)
im.save("cropped_images_database/" + filename)

# convert images to 128 encoding vectors

#database = {}
db = np.load('database.npy')
database=db[()]
key = new_name
database.setdefault(key, [])
database[key].append(img_to_encoding(dst, FRmodel))

print("key", key)

# accessing values under the same key
#database['jeon'][0][0]
#database['jeon'][1][0]

"""
10 encodings per person at maximum.
"""        

# save encoding vectors; 
np.save('database.npy', database)

print('database',database)

database['jeon']
