#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:56:45 2019

@author: youngjinlee
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
import os
import time
from gtts import gTTS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 

from itertools import chain

"""
LOAD PRETRAINED MODEL
"""

# Define FRmodel
FRmodel = faceRecoModel(input_shape=(3, 96, 96))# bottleneck
print('d')
# Loading pre-trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print('a')
load_weights_from_FaceNet(FRmodel) # bottleneck(critical)
print("Total Params:", FRmodel.count_params())


database = np.load('database.npy')
database
db=database[()]
print(db.keys())
len(db['bdt_yjlee_920717'])
len(db['bdt_malee_950523'])
len(db['bdt_rhlee_930517'])
len(db['bdt_chlee_930520'])
len(db['bdt_hwpark_950101'])
len(db['bdt_sjjeon_870101'])

del db['bdt_rhlee_930517']
del db['bdt_yjlee_920717']
del db['bdt_chlee_930520']


db['bdt_hwpark_950101'].pop(14)
db['bdt_hwpark_950101'].pop(7)
db['bdt_hwpark_950101'].pop(3)

db['bdt_sjjeon_870101'].pop(7)
db['bdt_sjjeon_870101'].pop(16)
db['bdt_sjjeon_870101'].pop(0)
db['bdt_sjjeon_870101'].pop(0)


temp = pd.DataFrame(db)
transposed_temp = temp.T 
unstacked = transposed_temp.unstack().reset_index()
unstacked.columns = ['num', 'face_id', 'value']
unstacked['value'] = list(chain.from_iterable(unstacked['value']))
unstacked['value'] = unstacked['value'].astype(str)
unstacked['value'] = unstacked['value'].map(lambda x: x.lstrip('[').rstrip(']'))
encoding = pd.DataFrame(unstacked['value'].str.split().values.tolist())
df = pd.concat([unstacked, encoding], axis=1)
df = df.drop(columns=['value'])
df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric)

"""
Data Preparation
"""
X = np.array(df.iloc[:,2:])
Y = df['face_id']
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, random_state = 2) 

X_all = np.vstack((X_train,X_test))
y_all = np.hstack((y_train,y_test))

encoded_Y
Y
label_dict = {0: "bdt_hwpark_950101",
              1: "bdt_malee_950523",
              2: "bdt_sjjeon_870101"}

"""
SVM
"""
# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 2).fit(X_train, y_train)
svm_model_linear_all = SVC(kernel = 'linear', C = 2).fit(X_all, y_all) 
svm_predictions = svm_model_linear.predict(X_test) 
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
cm
accuracy

"""
kNN
"""
# training a KNN classifier 
knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train) 
knn_all = KNeighborsClassifier(n_neighbors = 1).fit(X_all, y_all) 
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
cm
accuracy

"""
Naive bayes Classifier
"""
# training a Naive Bayes classifier 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_all = GaussianNB().fit(X_all, y_all) 
gnb_predictions = gnb.predict(X_test)   
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test)   
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 
cm
accuracy

db.keys()


db['bdt_malee_950523'].pop(14)
db['bdt_malee_950523'].pop(6)
db['bdt_malee_950523'].pop(1)
db['bdt_malee_950523'].pop(1)
db['bdt_malee_950523'].pop(10)
db['bdt_malee_950523'].pop(10)
db['bdt_malee_950523'].pop(0)
db['bdt_malee_950523'].pop(11)
db['bdt_malee_950523'].pop(7)
db['bdt_malee_950523'].pop(15)

db['bdt_hwpark_950101'].pop(12)
db['bdt_hwpark_950101'].pop(14)
db['bdt_hwpark_950101'].pop(20)
db['bdt_hwpark_950101'].pop(22)
db['bdt_hwpark_950101'].pop(18)

db['bdt_sjjeon_870101'].pop(5)

"""
main code
"""

pics_path = '/Volumes/sh/pics/'
mp3_path_from = "/Users/youngjinlee/Desktop/bdt_project_fr/audio/"
mp3_path_to = "/Volumes/sh/audio/"

while True:
    lst = []
    for item in os.listdir(pics_path):
        if not item.startswith('.') and os.path.isfile(os.path.join(pics_path, item)):
            if item == 'camera_test.py':
                pass
            else:
                comp = int(item.split('.')[0])
                lst.append(comp)
    print(len(lst))
    time.sleep(0.5)
    
    if len(lst) > 10:
        time.sleep(4.8)
        print(time.ctime())
        lst = sorted(lst)
        
        filename = str(lst[len(lst)-1]) + ".jpg"
        filename
        
        image = cv.imread(pics_path + filename, 1)# load the input image and convert it from BGR to RGB
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("[INFO]detecting faces...")
        boxes = face_recognition.face_locations(rgb, model="cnn") # bottleneck, unavoidable
        
        largest = 0
        for i in range(len(boxes)):
            current = (boxes[i][2] - boxes[i][0]) * (boxes[i][1] - boxes[i][3])
            print(current)
            if current > largest:
                largest = current
                j = i
                print(j)
        print('num of faces detected', len(boxes))
        
        if len(boxes) != 0:
        
            box = boxes[j]
    
            crop_img = image[box[0]:box[2], box[3]:box[1]]
            
            dst = cv.resize(crop_img, dsize=(96,96), interpolation=cv.INTER_AREA)
            
            im = Image.fromarray(dst)
            im.save("cropped_images/" + filename)
            
            encoding = img_to_encoding(dst, FRmodel)
            
            min_dist = 100
            for (name, lst) in db.items():
                print(name)
                for db_enc in lst:
                    dist = np.linalg.norm(db_enc - encoding)
                    print(dist)
                    if dist < min_dist:
                        min_dist = dist
                        identity = name
                        
            if min_dist > 0.53:
                textforspeech = "Not in the database."
                print("Not in the database" + " " + "The min dist is " + str(min_dist))
                shutil.copy(mp3_path_from + "not_verified.mp3", mp3_path_to + "tts.mp3")
            else:
                svm_pred = svm_model_linear_all.predict(encoding) 
                knn_pred = knn_all.predict(encoding)
                gnb_pred = gnb_all.predict(encoding)
                unique = np.hstack((svm_pred, knn_pred, gnb_pred))
                
                if len(np.unique(unique)) != 3:
                    res = np.argmax(np.bincount(unique))
                    identity2 = label_dict[res]
                    print("it's " + str(identity2))
                    textforspeech = "it's " + str(identity2) + ", the distance is " + str(min_dist)
                    shutil.copy(mp3_path_from + identity2 +".mp3", mp3_path_to + "tts.mp3")
                else:
                    print ("it's " + str(identity) + ", the distance is " + str(min_dist))
                    textforspeech = "it's " + str(identity) + ", the distance is " + str(min_dist)
                    shutil.copy(mp3_path_from + identity +".mp3", mp3_path_to + "tts.mp3")
                    
                print("Identity: " + str(identity) + ", the distance is " + str(min_dist))
                print("Identity2: " , str(identity2))
                print(unique)
                
                