# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:58:24 2019

@author: Cao Liang
"""

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import pandas as pd
import datetime as dt
import h5py
import cv2
import os

from glob import glob 
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def read_img_h5py(h5_file='data.h5'):
    data_set = None
    lbl_set = None
    test_index = 0
    
    with h5py.File(h5_file, 'r') as hf:
              
        start_index = int(hf['start_index'].value)
        end_index = int(hf['end_index'].value)
        print('start_index:', start_index, 'end_index:', end_index)
            
        for i in range(start_index, end_index):
            test_index += 1
            
            img_raw = hf['X'+str(i)]
            img_data = np.array(img_raw[:,:,:])
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            
            img_data = img_data.reshape(-1, 128, 128, 3)
            
            img_lbl = np.array(hf['y'+str(i)].value)
            
            img_id = np.array(hf['id'+str(i)].value)
            
            if data_set is None:
                data_set = img_data
                lbl_set = img_lbl
                id_set = img_id
            else:
                data_set = np.concatenate((data_set, img_data))
                lbl_set = np.concatenate((lbl_set, img_lbl))
                id_set = np.concatenate((id_set, img_id))
            
            if test_index % 1000 == 0:
                print('img data shape:', img_data.shape)
                print('data_set shape', data_set.shape)
                print('img_lbl:', img_lbl)
                print('lbl_set shape:', lbl_set.shape)
                print('img_id:', img_id)
                print('id_set shape:', id_set.shape)
                
        
    return (data_set, lbl_set, id_set) 

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
    data_map = { 'cat':0, 'bird':1, 'dog':2 }
    
    with h5py.File(h5_file, 'r') as hf:
        X_train = hf['X_train'].value
        
        y_train = pd.Series(hf['y_train']).map(data_map)
        print('Read y_train: ', y_train.shape)        
        
        X_test = hf['X_test'].value
        
        y_test = pd.Series(hf['y_test']).map(data_map)
        
    return (X_train, y_train, X_test, y_test)


h5_file = 'img_128.h5'
X_test_data, y_test_data, img_id_data = read_img_h5py(h5_file=h5_file)
print('Read X_test_data: ', X_test_data.shape)        
print('Read y_test_data: ', y_test_data.shape)        
print('Read img_id_data: ', img_id_data.shape)        
    
(tsDat, tsLbl)  = X_test_data, y_test_data
print('Read tsDat: ', tsDat.shape)

                            # Convert the data into 'float32'
                            # Rescale the values from 0~255 to 0~1
tsDat       = tsDat.astype('float32')/255


                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = tsDat.shape[1]
imgclms     = tsDat.shape[2]
channel     = tsDat.shape[3]


                            # Perform one hot encoding on the labels
                            # Retrieve the number of classes in this problem
tsLbl       = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

                            # fix random seed for reproducibility
seed        = 42
np.random.seed(seed)

optmz       = optimizers.Adam(lr=0.0001)

#optmz       = optimizers.RMSprop(lr=0.0002, rho=0.9, 
#                                 epsilon=1e-08, decay=0.01/4000)

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
                  kernel_initializer='he_normal',
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
  v = resBlkV1(inputs=v,
              numFilters=16,
              numBlocks=7,
              downsampleOnFirst=False,
              names='Stg1')
  v = resBlkV1(inputs=v,
              numFilters=32,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg2')
  v = resBlkV1(inputs=v,
              numFilters=64,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg3')
  v = resBlkV1(inputs=v,
              numFilters=128,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg4')
  v = AveragePooling2D(pool_size=8,
                      name='AvgPool')(v)
  v = Flatten()(v)
  outputs = Dense(numClasses,
                 activation='softmax',
                 kernel_initializer='he_normal')(v)
  model = Model(inputs=inputs,outputs=outputs)
  model.compile(loss='categorical_crossentropy',
               optimizer=optmz,
               metrics=['accuracy'])
    
  return model

def save_img_id_list(out_file='list.txt', img_id_list=[]):
    if img_id_list is None or len(img_id_list) == 0:
        print('img_id_list empty, so not save')
        return
    
    with open(out_file, 'w') as of:
        for line in img_id_list:
            of.write(line)
            of.write('\n')


                                # Setup the models
modelGo     = createResNetV1()  # This is used for final testing
modelGo.summary()

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy

model_trained = 'PRMLS_CA2_74.54'
filepath        = model_trained + ".hdf5"
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
print("predicts: ", predicts.shape)
print(predicts[:3, :3])
print("predicts max: ", predicts.max())
                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
print("predout: ", predout.shape)

testout     = np.argmax(tsLbl,axis=1)
print("testout: ", testout.shape)

labelname   = ['cat',
               'bird',
               'dog']
                                            # the labels for the classfication report

testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)

testout_df_data = img_id_data.reshape([len(img_id_data), 1])
testout_df_data = np.concatenate((testout_df_data, tsLbl), axis=1)
testout_df_data = np.concatenate((testout_df_data, predicts), axis=1)
print('testout_df_data: ', testout_df_data.shape)

testout_df = pd.DataFrame(data=testout_df_data, 
                          columns=['img_id', 'img_cat', 'img_bird', 'img_dog', 
                                   'cat', 'bird', 'dog'])
print('testout_df: ', testout_df.shape)
#testout_df.set_index('img_id', inplace=True)
print(testout_df.head(n=3))

testout_df['prob'] = 0
testout_cols = list(testout_df.columns)
print('testout_cols: ', testout_cols)
print(testout_df.head(n=3))

testout_df.loc[testout_df.img_cat == 1, 'prob'] = testout_df['cat']
testout_df.loc[testout_df.img_bird == 1, 'prob'] = testout_df['bird']
testout_df.loc[testout_df.img_dog == 1, 'prob'] = testout_df['dog']

testout_df = testout_df.sort_values(by=['prob'], ascending=True)
print('testout_df: ', testout_df.shape)

unfit_threshold = 0.1

unfit_img_df = testout_df[testout_df.prob < unfit_threshold]
print("unfit_img_df: ", unfit_img_df.shape)

fit_img_df = testout_df[testout_df.prob >= unfit_threshold]
print("fit_img_df: ", fit_img_df.shape)

fit_df = pd.DataFrame(data=[['unfit img', unfit_img_df.shape[0]], 
                            ['fit img', fit_img_df.shape[0]]],
                      columns=['Image Data', 'Quantity'])
#print(fit_df['label'])

ax = fit_df.plot(kind='bar', x='Image Data', y='Quantity', rot=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), 
                (p.get_x() * 1.005, p.get_height() * 1.005))

# Save unfit image id list to file
unfit_img_id_list = list(unfit_img_df['img_id'].values)
print('unfit_img_id_list: ', len(unfit_img_id_list))
unfit_img_id_list_file = '..\\unfit_list.txt'
save_img_id_list(out_file=unfit_img_id_list_file, 
                 img_id_list=unfit_img_id_list)
print('Save unfit image id list to file:', unfit_img_id_list_file)
