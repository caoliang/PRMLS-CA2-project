# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:58:24 2019

@author: Jacky
@author: Cao Liang
"""

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import h5py

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    
                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

def read_data_set(h5_file='out.h5'):
    with h5py.File(h5_file, 'r') as hf:
        X_train = hf['X_train'].value
        print('Read X_train: ', X_train.shape)        
        
        y_train = hf['y_train'].value
        print('Read y_train: ', y_train.shape)        
        
        X_test = hf['X_test'].value
        print('Read X_test: ', X_test.shape)        
        
        y_test = hf['y_test'].value
        print('Read y_test: ', y_test.shape)        
    
    return (X_train, y_train, X_test, y_test)

X_train_data, y_train_data, X_test_data, y_test_data = read_data_set(h5_file='ca2data.h5' )

#data            = cifar10.load_data()
(trDat, trLbl)  = X_train_data, y_train_data
(tsDat, tsLbl)  = X_test_data, y_test_data


                            # Convert the data into 'float32'
                            # Rescale the values from 0~255 to 0~1
trDat       = trDat.astype('float32')/255
tsDat       = tsDat.astype('float32')/255


                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = trDat.shape[1]
imgclms     = trDat.shape[2]
channel     = trDat.shape[3]


                            # Perform one hot encoding on the labels
                            # Retrieve the number of classes in this problem
trLbl       = to_categorical(trLbl)
tsLbl       = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

                            # fix random seed for reproducibility
seed        = 42
np.random.seed(seed)

optmz       = optimizers.Adam(lr=0.00005)
modelname   = 'PRMLS_CA2'
                            # define the deep learning model

def resLyr(inputs,
           numFilters=16,
           kernelSz=3,
           strides=1,
           activation='relu',
           batchNorm=True,
           convFirst=True,
           lyrName=None):
  convLyr = Conv2D(numFilters,
                  kernel_size=kernelSz,
                  strides=strides,
                  padding='same',
                  kernel_initializer=he_normal(33),
                  kernel_regularizer=l2(1e-4),
                  name=lyrName+'_conv' if lyrName else None)
  x = inputs
  if convFirst:
    x = convLyr(x)
    if batchNorm:
      x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
    if activation is not None:
      x = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
  else:
    if batchNorm:
      x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
    if activation is not None:
      x = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
    x = convLyr(x)

  return x


def resBlkV1(inputs,
             numFilters=16,
             numBlocks=5,
             downsampleOnFirst=True,
             names=None):
  x = inputs
  for run in range(0,numBlocks):
    strides = 1
    blkStr = str(run+1)
    if downsampleOnFirst and run == 0:
      strides = 2
    y = resLyr(inputs=x,
              numFilters=numFilters,
              strides=strides,
              lyrName=names+'_Blk'+blkStr+'_Res1' if names else None)
    y = resLyr(inputs=y,
              numFilters=numFilters,
              activation=None,
              lyrName=names+'_Blk'+blkStr+'_Res2' if names else None)
    if downsampleOnFirst and run == 0:
      x = resLyr(inputs=x,
                numFilters = numFilters,
                kernelSz=1,
                strides=strides,
                activation=None,
                batchNorm=False,
                lyrName=names+'_Blk'+blkStr+'_lin' if names else None)
    x = add([x,y],
           name=names+'_Blk'+blkStr+'_add' if names else None)
    x  = Activation('relu',
                   name=names+'_Blk'+blkStr+'_relu' if names else None)(x)
     
  return x
    
def createResNetV1(inputShape=(128,128,3),
                        numClasses=3):
  inputs = Input(shape=inputShape)
  v = resLyr(inputs,
             lyrName='Input')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=16,
              numBlocks=3,
              downsampleOnFirst=False,
              names='Stg1')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=32,
              numBlocks=3,
              downsampleOnFirst=True,
              names='Stg2')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=64,
              numBlocks=3,
              downsampleOnFirst=True,
              names='Stg3')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=128,
              numBlocks=3,
              downsampleOnFirst=True,
              names='Stg4')
  
  v = Dropout(0.2)(v)

  v = resBlkV1(inputs=v,
              numFilters=256,
              numBlocks=3,
              downsampleOnFirst=True,
              names='Stg5')

  v = Dropout(0.2)(v)
  
  v = AveragePooling2D(pool_size=8,
                      name='AvgPool')(v)
  
  v = Dropout(0.2)(v)
  
  v = Flatten()(v)

  outputs = Dense(numClasses,
                 activation='softmax',
                 kernel_initializer=he_normal(33))(v)
  
  model = Model(inputs=inputs,outputs=outputs)
  model.compile(loss='categorical_crossentropy',
               optimizer=optmz,
               metrics=['accuracy'])
    
  return model
                                # Setup the models
modelGo     = createResNetV1()  # This is used for final testing

modelGo.summary()

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + "_testing.hdf5"

                            # Log the epoch detail into csv

                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer=optmz, 
                metrics=['accuracy'])

                            # Make classification on the test dataset
predicts    = modelGo.predict(tsDat)


                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(tsLbl,axis=1)
labelname   = ['cat',
               'bird',
               'dog']
                                            # the labels for the classfication report

testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)
