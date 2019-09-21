#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:28:19 2019

@author: youngjinlee

@title: image analysis using tsne
"""


import numpy as np
from itertools import chain
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns

database = np.load("database_prune.npy")
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

#불러온 db 
temp1_df = df.loc[df['face_id'] == 'bdt_malee_950523',:]
temp2_df =df.loc[df['face_id'] == 'bdt_hwpark_950101',:]
temp3_df =df.loc[df['face_id'] == 'bdt_sjjeon_870101',:]
temp_df = pd.concat([temp1_df,temp2_df,temp3_df])#malee, hwpark,sjjeon 데이터 병합
temp4_df = temp_df.iloc[0:,2:]#3명이 합쳐진 데이터에서 벡터값만 불러오기
temp4_df.reset_index(drop=True,inplace=True)#인덱스 reset
temp_df.reset_index(drop=True,inplace=True)
face_matrix=temp4_df.iloc[:33,0:127].values

#난수 생성 범위 고정
np.random.seed(0)

#tsne에 활용 가능하도록 데이터 변형
face_tsne_result=TSNE(learning_rate=200).fit_transform(face_matrix) 
#tsne에 맞게 변형된 데이터에 원래 레이블 맵핑
df_face_tsne_result = (
  pd.DataFrame(face_tsne_result, columns=['V1', 'V2'])
   .assign(face_id = temp_df['face_id'])
)
df=df_face_tsne_result

#plot 그리기
sns.lmplot(x='V1',y='V2',data=df,fit_reg=False,hue='face_id',legend=False) 
plt.legend(bbox_to_anchor=(1.1, 1.5))#label 
plt.show()
