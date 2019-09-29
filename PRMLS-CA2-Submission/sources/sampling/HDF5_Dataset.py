# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:53:08 2019

@author: cl
"""

import cv2
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

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
            
            if data_set is None:
                data_set = img_data
                lbl_set = img_lbl
            else:
                data_set = np.concatenate((data_set, img_data))
                lbl_set = np.concatenate((lbl_set, img_lbl))
            
            if i % 100 == 0:
                print('img data shape:', img_data.shape)
                print('data_set shape', data_set.shape)
                print('img_lbl:', img_lbl)
                print('lbl_set shape:', lbl_set.shape)
    
    return (data_set, lbl_set)    

def write_data_set(X_train, X_test, y_train, y_test, h5_file='out.h5'):
    dt = h5py.special_dtype(vlen=str)
    
    with h5py.File(h5_file, 'w') as hf:
        
        hf.create_dataset(name="X_train", shape=X_train.shape, dtype=np.int8,
                          compression="gzip", compression_opts=9)
        hf['X_train'][...] = X_train
        print('Save X_train: ', X_train.shape)
        
        hf.create_dataset(name="y_train", shape=y_train.shape, 
                          dtype=dt,
                          compression="gzip", compression_opts=9)
        hf['y_train'][...] = y_train
        print('Save y_train: ', y_train.shape)
        
        hf.create_dataset(name="X_test", shape=X_test.shape, dtype=np.int8,
                          compression="gzip", compression_opts=9)
        hf['X_test'][...] = X_test
        print('Save X_test: ', X_test.shape)
        
        hf.create_dataset(name="y_test", shape=y_test.shape, 
                          dtype=dt,
                          compression="gzip", compression_opts=9)
        hf['y_test'][...] = y_test
        print('Save y_test: ', y_test.shape)
        
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
        
(X, y) = read_img_h5py(h5_file='..\\data\\data128.h5')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=1 / 3, 
                                                    random_state=1)
print('X_train: ', X_train.shape, 'X_test: ', X_test.shape)
print('y_train: ', y_train.shape, 'y_test: ', y_test.shape)

write_data_set(X_train, X_test, y_train, y_test, h5_file='..\\data\\ca2data.h5')

X_train_data, y_train_data, X_test_data, y_test_data = read_data_set(h5_file='..\\data\\ca2data.h5' )
