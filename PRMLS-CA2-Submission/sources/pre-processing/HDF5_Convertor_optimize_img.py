# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:27:09 2019

@author: Jacky
"""

import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
from glob import glob

def proc_images(img_path='dt_cat', img_name='cat', 
                img_ext='png', out_file="data.h5",
                start_index=1, img_label=0):
    """
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class label
    """
    start = dt.datetime.now()
    # ../input/
    #PATH = os.path.abspath(os.path.join('..', 'input'))
    # ../input/sample/images/
    #SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
    # ../input/sample/images/*.png
    #images = glob(os.path.join(SOURCE_IMAGES, "*.png"))
    images = glob(os.path.join(img_path, "*" + img_ext))
    
    # Load labels
    #labels = pd.read_csv('../input/sample_labels.csv')
    # Get all image files
    img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
        
    # Size of data
    NUM_IMAGES = len(images)
    HEIGHT = 128
    WIDTH = 128
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    with h5py.File(out_file, 'a') as hf:
        img_index = start_index
        img_end_index = start_index
        
        for i,img in enumerate(images):
            if img_index > start_index:
                img_end_index = img_index
            
            # Images
            image = cv2.imread(img)
            image = cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
            Xset = hf.create_dataset(
                name='X'+str(img_index),
                data=image,
                shape=(HEIGHT, WIDTH, CHANNELS),
                maxshape=(HEIGHT, WIDTH, CHANNELS),
                compression="gzip",
                compression_opts=9)
            yset = hf.create_dataset(
                name='y'+str(img_index),
                data=img_label,
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
            img_id = '{0}_{1}'.format(img_name, os.path.basename(img))
            idset = hf.create_dataset(
                name='id'+str(img_index),
                data=img_id,
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
            end=dt.datetime.now()
            
            if img_index % 100 == 0:
                print(img_index, ": ", img_name, ", ", (end-start).seconds, "seconds")
            img_index += 1
            
        return img_end_index

def store_total_img_indexes(out_file='data.h5', start_index=0, end_index=0):
    with h5py.File(out_file, 'a') as hf:
        hf.create_dataset(
                name='start_index',
                data=str(start_index),
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
        print('Store start index', start_index)

        hf.create_dataset(
                name='end_index',
                data=str(end_index),
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
        print('Store end index', end_index)


img_start_index=0
img_end_index=0

h5_file = 'img_128.h5'
start_processing = dt.datetime.now()

if os.path.exists(h5_file):
    os.remove(h5_file)

img_end_index = proc_images(img_path='dt_cat', img_name='cat', img_ext='jpg', 
            out_file=h5_file, start_index=img_start_index, img_label=0)
print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

img_start_index = img_end_index + 1

img_end_index = proc_images(img_path='dt_bird', img_name='bird', img_ext='jpg', 
            out_file=h5_file, start_index=img_start_index, img_label=1)
print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

img_start_index = img_end_index + 1
img_end_index = proc_images(img_path='dt_dog', img_name='dog', img_ext='jpg', 
            out_file=h5_file, start_index=img_start_index, img_label=2)

print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

store_total_img_indexes(out_file=h5_file, start_index=0, end_index=img_end_index)

end_processing = dt.datetime.now()
print('Total spent: ', (end_processing - start_processing))

#!ls -lha

with h5py.File(h5_file, 'r') as hf:
    plb.imshow(hf["X2383"])
    print(hf["y2383"].value)
    print(hf["id2383"].value)
    print(hf['start_index'].value)
    print(hf['end_index'].value)


