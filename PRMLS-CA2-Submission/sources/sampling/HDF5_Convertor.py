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
import sys
import pandas as pd
from glob import glob
import shutil

def read_unfit_img_id_list(id_list_file='img_id.txt'):
    img_id_map = {}
    img_id_list = None
    with open(file=id_list_file, mode='r') as id_file:
        img_id_list = id_file.readlines()
    
    if img_id_list is None:
        print('img id list file is empty')
        return img_id_map
    
    for img_id in img_id_list:
        img_id = '' if img_id is None else img_id.strip()
        if img_id != '':
            img_id_map[img_id] = ''
    
    return img_id_map

def proc_images(img_path='dt_cat', img_name='cat', 
                img_ext='png', out_file="data.h5",
                start_index=1, img_label=0, unfit_id_map={},
                unfit_img_folder='unfit_img'):
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
    HEIGHT = 128
    WIDTH = 128
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    if not os.path.exists(unfit_img_folder):
        os.makedirs(unfit_img_folder)
    
    with h5py.File(out_file, 'a') as hf:
        img_index = start_index
        img_end_index = start_index
        
        for i,img in enumerate(images):
            if img_index > start_index:
                img_end_index = img_index
            
            # Images
            image = cv2.imread(img)
            image = cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
            
            img_id = '{0}_{1}'.format(img_name, os.path.basename(img))
            if img_id in unfit_id_map:
                print('Unfit image: ', img_id)
                
                # Copy unfit image to unfit image folder
                # adding exception handling
                try:
                    shutil.copy(img, unfit_img_folder)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                continue
                
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
                data=img_label,
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
            end=dt.datetime.now()
            
            if img_index % 100 == 0:
                print("\r", i, ": ", (end-start).seconds, "seconds", end="")
            
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


unfit_img_dir = '..\\pre-processing\\unfit_img'
unfit_img_list_map = read_unfit_img_id_list(id_list_file='unfit_list.txt')

img_start_index=0
img_end_index=0

out_h5_file = '..\\data\\data128.h5'
if os.path.exists(out_h5_file):
    os.remove(out_h5_file)

cat_img_path = '..\\pre-processing\\dt_cat'
img_end_index = proc_images(img_path=cat_img_path, img_name='cat', img_ext='jpg', 
            out_file=out_h5_file, start_index=img_start_index, img_label=0,
            unfit_id_map=unfit_img_list_map, unfit_img_folder=unfit_img_dir)
print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

img_start_index = img_end_index + 1

bird_img_path = '..\\pre-processing\\dt_bird'
img_end_index = proc_images(img_path=bird_img_path, img_name='bird', img_ext='jpg', 
            out_file=out_h5_file, start_index=img_start_index, img_label=1,
            unfit_id_map=unfit_img_list_map, unfit_img_folder=unfit_img_dir)
print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

img_start_index = img_end_index + 1

dog_img_path = '..\\pre-processing\\dt_dog'
img_end_index = proc_images(img_path=dog_img_path, img_name='dog', img_ext='jpg', 
            out_file=out_h5_file, start_index=img_start_index, img_label=2,
            unfit_id_map=unfit_img_list_map, unfit_img_folder=unfit_img_dir)

print('----------------')
print('start: {}, end: {}'.format(img_start_index, img_end_index))
print('----------------')

store_total_img_indexes(out_file=out_h5_file, start_index=0, 
                        end_index=img_end_index)

with h5py.File(out_h5_file, 'r') as hf:
    plb.imshow(hf["X2383"])
    print(hf["y2383"].value)
    print(hf['start_index'].value)
    print(hf['end_index'].value)


