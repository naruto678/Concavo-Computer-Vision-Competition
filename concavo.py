#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:34:39 2018

@author: arnab
"""
import numpy as np
import cv2 as cv
import os
def make_data():
    data=np.zeros((4500,128,128,3))
    labels=np.zeros((4500,1))
    count=0
    for i in range(15):
        filepath=str(i)+'/'
        filelist=os.listdir(str(i))
        for img in filelist[:300]:
            y=cv.imread(filepath+img)
            
            #y=cv.Canny(y,100,200)
            y=cv.resize(y,dsize=(128,128))/255
            
            data[count,:,:]=y
            labels[count,:]=i
            count=count+1
            
    return data,labels
data,labels=make_data()



from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(128, 128, 3))

from sklearn.model_selection import train_test_split
partial_X_train,val_X_train,partial_Y_train,val_Y_train=train_test_split(data,labels[:4500],test_size=0.2)
np.save('partial_X_train',partial_X_train)
np.save('partial_Y_train',partial_Y_train)
np.save('val_X_train',val_X_train)
np.save('val_Y_train',val_Y_train)


def vec(sequences,dimensions=15):
    sequences=sequences.astype('int')
    results=np.zeros((len(sequences),dimensions))
    for i,j in enumerate(sequences):
        results[i,j]=1
        
    return results
    
partial_Y_train=vec(partial_Y_train).astype('float64')
val_Y_train=vec(val_Y_train)
partial_X_train=conv_base.predict(partial_X_train)
val_X_train=conv_base.predict(val_X_train)



np.save('new_partial_X_train.npy',partial_X_train)
np.save('new_val_X_train.npy',val_X_train)


partial_X_train=partial_X_train.reshape((7872,8192))

dress_base=keras.applications.mobilenet.MobileNet(input_shape=(128,128,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
partial_X_train=dress_base.predict(partial_X_train)
val_X_train=dress_base.predict(val_X_train)
np.save('concavo partial_X_train.npy',partial_X_train)
 

def make_model():
    from keras import models
    from keras import layers
    model=models.Sequential()
     
    model.add(layers.Dense(1000,activation='relu',input_shape=(16384,)))
    model.add(layers.Dense(1000,activation='relu'))
    model.add(layers.Dense(600,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(15,activation='softmax'))
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

new_model=make_model()
new_model.fit(partial_X_train.reshape(3600,16384),partial_Y_train,epochs=15,batch_size=600,validation_data=(val_X_train.reshape(900,16384),val_Y_train))
#new_model.fit(partial_X_train.reshape(3600,128,128,1),partial_Y_train,epochs=5,batch_size=600,validation_data=(val_X_train.reshape(900,128,128,1),val_Y_train))
#
new_model.save('working_model',new_model)

results=[]

for count,img in enumerate(os.listdir('test')):
    y=np.zeros((1,128,128,3))
    y1=cv.imread('test/'+img)
    y[0,:,:,:]=cv.resize(y1,dsize=(128,128))/255
    y=dress_base.predict(y)
    results.append(np.argmax(new_model.predict(y.reshape(1,16384)))+1)
    print(count)
    
    



l1=os.listdir('test')


for i in range(21273):
    results[i]=results[i]+1





import pandas as pd
df=pd.DataFrame()
df['image_name']=l1
df['category']=results
df.to_csv('result1.csv')