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
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
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
        y_train = hf['y_train'].value
        
        X_test = hf['X_test'].value
        y_test = hf['y_test'].value
        
        X_val = np.concatenate((X_train, X_test))
        print('Read X_val: ', X_val.shape)        
        
        y_val = np.concatenate((y_train, y_test))
        print('Read y_val: ', y_val.shape)        
                
    return (X_val, y_val)

def createResNetV1_7631(inputShape=(128,128,3),
                        numClasses=3):
  inputs = Input(shape=inputShape)
  v = resLyr(inputs,
            lyrName='Input')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=16,
              numBlocks=7,
              downsampleOnFirst=False,
              names='Stg1')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=32,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg2')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=64,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg3')
  
  v = Dropout(0.2)(v)
  
  v = resBlkV1(inputs=v,
              numFilters=128,
              numBlocks=7,
              downsampleOnFirst=True,
              names='Stg4')
  
  v = Dropout(0.2)(v)
  
  v = AveragePooling2D(pool_size=8,
                      name='AvgPool')(v)
  
  v = Dropout(0.2)(v)
  
  v = Flatten()(v)
  outputs = Dense(numClasses,
                 activation='softmax',
                 kernel_initializer='he_normal')(v)
  model = Model(inputs=inputs,outputs=outputs)
  model.compile(loss='categorical_crossentropy',
               optimizer=optmz,
               metrics=['accuracy'])
    
  return model

# load models from file
def load_all_models(n_models, model_name='PRMLS_CA2_Ensemble_NN',
                    model_files=[], models_creation_func=[]):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        if len(model_files) == n_models:
            filename = model_files[i]
        else:
            filename = model_name + '_' + str(i + 1) + '.hdf5'
        
        # create model from model creation function list
        if len(models_creation_func) == n_models:
            model = models_creation_func[i]()
        else:
            model = createResNetV1_7631()
        
        # load model weight from file
        model.load_weights(filename)
        
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
 
# define stacked model from multiple member input models
def define_stacked_model(members, numClasses=3,
                         out_model_img=None,
                         model_optimizer='adam'):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # rename to avoid 'unique layer name' issue
            #print(layer.__dict__)
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name            
            # make not trainable
            layer.trainable = False

    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(len(members) * numClasses, activation='relu')(merge)
    output = Dense(numClasses, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    
    # plot graph of ensemble
    if out_model_img is not None:
        plot_model(model, to_file=out_model_img, show_shapes=True, 
                   show_layer_names=True)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, 
                  metrics=['accuracy'])
    return model

# ----------------------------------
# All files pathes are defined below
# ----------------------------------
# Number of models
n_members = 2
# Ensemble model name
modelname   = 'PRMLS_CA2_Ensemble_NN'
# Images data hdf5 file
img_data_h5_file = '..\\ca2data.h5'
# model hdf5 files
model_files_list = [ '..\\models\\' + modelname + '_1.hdf5', 
                     '..\\models\\' + modelname + '_2.hdf5' ]
# model creation function list
model_create_list = [ createResNetV1_7631, createResNetV1_7631 ]
# Ensemble model hdf5 file
filepath        = '..\\models\\' + modelname + ".hdf5"


X_test_data, y_test_data = read_data_set(h5_file=img_data_h5_file)
print('Read X_test_data: ', X_test_data.shape)        
print('Read y_test_data: ', y_test_data.shape)        
    
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
optmz       = optimizers.Adam(lr=0.0001)


                            # define the deep learning model
# load all models
members = load_all_models(n_members, model_files=model_files_list,
                          models_creation_func=model_create_list)
print('Loaded %d models' % len(members))

X_tsDat = [tsDat for model in members]
print("X_tsDat: ", np.array(X_tsDat).shape)

stacked_model_go = define_stacked_model(members, numClasses=num_classes,
                                        model_optimizer=optmz)                        
stacked_model_go.load_weights(filepath)
print('Loaded stacked model')


                            # Make classification on the test dataset
predicts    = stacked_model_go.predict(X_tsDat)
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

testout_df_data = np.concatenate((tsLbl, predicts), axis=1)
print('testout_df_data: ', testout_df_data.shape)

testout_df = pd.DataFrame(data=testout_df_data, 
                          columns=['img_cat', 'img_bird', 'img_dog', 
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

testout_df['class_name'] = ''
testout_df.loc[testout_df.img_cat == 1, 'class_name'] = 'cat'
testout_df.loc[testout_df.img_bird == 1, 'class_name'] = 'bird'
testout_df.loc[testout_df.img_dog == 1, 'class_name'] = 'dog'
print(testout_df.head(n=3))

testout_df = testout_df.sort_values(by=['prob'], ascending=True)
print('testout_df: ', testout_df.shape)

fit_step = 0.1
from_fit = 0
to_fit = 0
fit_count_list = []
fit_count_lbl = []
fit_count_class = []

fit_list = []
class_names = ['cat', 'bird', 'dog']
row_index = 0
    
for i in np.arange(0, 1, fit_step):
    from_fit = i
    to_fit = from_fit + fit_step
    
    fit_range = '{0:.1f}-{1:.1f}'.format(from_fit, to_fit)
    fit_img_df = testout_df[testout_df['prob'] > from_fit]
    fit_img_df = fit_img_df[testout_df['prob'] <= to_fit]
        
    for class_n in class_names:
        fit_class_df = fit_img_df[fit_img_df['class_name'] == class_n]
        fit_count = fit_class_df.shape[0]
        fit_row = {'quantity': fit_count, 'range': fit_range,
                    'class_name': class_n}
        print('add fit_row: ', row_index)
        fit_list.append(fit_row)
        print(len(fit_list))
        row_index += 1

fit_df = pd.DataFrame(fit_list)
print(fit_df.shape)
color_list = ['#CC00CC', '#0000BB', 'green']
              
for i, class_n in enumerate(class_names):
    fit_class_df = fit_df[fit_df['class_name'] == class_n]
    chart_title = '{} Image Data'.format(class_n.capitalize())
     
    ax = fit_class_df.plot(kind='bar', x='range', y='quantity', 
                     rot=0, colors=color_list[i], title=chart_title,
                     figsize=(10,5))
    for p in ax.patches:
        ax.annotate(str(p.get_height()), 
                    (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()

