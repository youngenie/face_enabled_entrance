#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:54:17 2019

@author: youngjinlee

@title: multi class classification with Keras
"""
import numpy as np
import pandas as pd
from itertools import chain

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
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


database = np.load('database.npy')
database
db=database[()]
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


"""
SVM
"""
# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 2).fit(X_train, y_train) 
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
knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train) 
  
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
gnb_predictions = gnb.predict(X_test)   
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test)   
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 
cm
accuracy



"""
MLP KERAS
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random 
import time 
from sklearn import preprocessing, model_selection
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


mlp_X = np.array(df.iloc[:,2:])
mlp_Y = df['face_id']

encoder = LabelEncoder()
encoder.fit(mlp_Y)
mlp_Y = encoder.transform(mlp_Y)

#train_x, test_x, train_y, test_y = model_selection.train_test_split(mlp_X,mlp_Y,test_size = 0.2, random_state = 0)
#train_x, test_x, train_y, test_y = model_selection.train_test_split(mlp_X,mlp_Y, random_state = 0)
train_x, test_x, enc_train_y, enc_test_y = train_test_split(mlp_X, mlp_Y, random_state = 2) 

train_Y = np_utils.to_categorical(enc_train_y)
test_Y = np_utils.to_categorical(enc_test_y)

enc_test_y.shape
y_test.shape
test_y.shape


input_dim = 128
numppl = 4

model = Sequential()
model.add(Dense(2, input_dim = input_dim , activation = 'sigmoid'))
#model.add(Dense(10, activation = 'relu'))
model.add(Dense(numppl, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 10, batch_size = 2)
scores = model.evaluate(test_x, test_y)
scores
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

prediction
predictions = model.predict_classes(test_x)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)
for i, j in zip(prediction_ , test_y):
    print( " the nn predict {}, and the species to find is {}".format(i,j))

predictions
test_y
cm = confusion_matrix(enc_test_y, predictions) 
cm
