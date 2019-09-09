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
                start_index=1):
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
    labels = pd.DataFrame({'image_file': img_files})
    labels['labels'] = img_name   
        
    # Size of data
    NUM_IMAGES = len(images)
    HEIGHT = 256
    WIDTH = 256
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    with h5py.File(out_file, 'a') as hf:
        img_index = start_index
        
        for i,img in enumerate(images):            
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
            # Labels
            base = os.path.basename(img)
            #finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
            yset = hf.create_dataset(
                name='y'+str(img_index),
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
            end=dt.datetime.now()
            print("\r", i, ": ", (end-start).seconds, "seconds", end="")
            img_index += 1
            
        return img_index

img_start_index = proc_images(img_path='dt_cat', img_name='cat', img_ext='jpg', 
            out_file="data.h5", start_index=1)
img_start_index = proc_images(img_path='dt_dog', img_name='dog', img_ext='jpg', 
            out_file="data.h5", start_index=img_start_index)
img_start_index = proc_images(img_path='dt_bird', img_name='bird', img_ext='jpg', 
            out_file="data.h5", start_index=img_start_index)
#proc_images(img_path='dt_dog', img_name='dog', img_ext='jpg', out_file="data.h5")
#proc_images(img_path='dt_bird', img_name='bird', img_ext='jpg', out_file="data.h5")

#!ls -lha

with h5py.File('data.h5', 'r') as hf:
    plb.imshow(hf["X3000"])
    print(hf["y2"].value)