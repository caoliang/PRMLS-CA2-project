# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:07:32 2019

@author: Jacky
@author: Cao Liang
"""

from pypexels import PyPexels
from urllib.error import HTTPError
import requests
from datetime import datetime
import time
import os

def show_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def get_hours_spent(start_time):
    return int((time.time() - start_time) / 3600)

def show_time_spent(start_time):
    time_used = time.time() - start_time
    return "{}:{}:{}".format(int(time_used / 3600), int(time_used % 3600 / 60), int(time_used % 60))


def save_image(folder_name, file_name, file_url):
    file_full_path = os.path.join(folder_name, file_name)
    
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
                
        url_resp = requests.get(file_url, allow_redirects=True)
        
        with open(file_full_path, "wb") as img_file:
            img_file.write(url_resp.content)
        
        #urlretrieve(file_url, file_full_path)
        return True
    except FileNotFoundError as err:
        print(err)   # something wrong with local path
    except HTTPError as err:
        print(err)

    return False

def download_images(py_api, search_words, ignore_num=0, num_taget=190):
    search_results = py_pexels.search(query=search_words, per_page=100)
    num_found = 0
    
    while search_results is not None:
        print('Search results {0} items, target {1} itmes'.format(
                search_results.total_results, num_taget))
        print('Current page: {}'.format(search_results.page))
        for photo in search_results.entries:
            if num_found < ignore_num:
                pass
                # Ignore image files
                #print('Skip image {0}'.format(photo.id))
            else:
                #print('src: ', photo.src)
                photo_url = photo.src.get('medium')
                photo_ext = photo_url.split('?')[0].split('.')[-1]
                photo_name = '{0}.{1}'.format(photo.id, photo_ext)
                #print(photo.id, photo.photographer, photo_url)
                result = save_image(search_words, photo_name, photo_url)
                print('Save image: {0}, result: {1}'.format(photo_name, result))
                
                if not result:
                    raise Exception('Cannot retrieve image {0}'.format(photo_url))
                    break

            # Increase number of images found in search results            
            num_found += 1
            #print('Found {0} images'.format(num_found))
            
            if num_found >= num_taget:
                print('Found {0} images as plan'.format(num_found))
                break
            

        if not search_results.has_next:
            print('Next page result not found')
            break
            
        if num_found >= num_taget:
            print('Found {0} images as target'.format(num_found))
            break
        
        search_results = search_results.get_next_page()
        if search_results:
            print('Next page: {}'.format(search_results.page))
        else:
            print('Next page not found')
       
    return num_found

def complete_download_task(py_api, search_words, target_num_images,
                           ignore_num_images=0):
    total_downloaded = 0
    time_start_dl = time.time()
    
    if ignore_num_images > 0:
        num_download_plan = target_num_images + ignore_num_images
    else:
        num_download_plan = target_num_images
        
    print('Try to download {0} images, {1} ignored'.format(
            num_download_plan, ignore_num_images))
        
    # Download images within 190 in 1 hour
    num_found = download_images(py_api, search_words,
                                     ignore_num=ignore_num_images,
                                     num_taget=num_download_plan)
    if num_found > ignore_num_images:
        total_downloaded = num_found - ignore_num_images
    else:
        total_downloaded = 0
        
    num_remaining = target_num_images - total_downloaded
    print('Downloaded {0} images, {1} remaining, time spent: {2}'.format(
            total_downloaded, num_remaining, show_time_spent(time_start_dl)))
    
    # Need to complete download, and try next time again manually
    print('....')
    print('Completed download')
    #time.sleep(30)
    print('....')
    
    return total_downloaded

# api_key from Cao Liang
#api_key = '563492ad6f917000010000018cd4391b02e24cb683cd90ac888d7189'
# api_key from Jacky
api_key = '563492ad6f917000010000015dee3b636ab44279b6f1edf055de1fe0'

# instantiate PyPexels object
py_pexels = PyPexels(api_key=api_key)

search_target = 'domestic cat'
# Each time only allow maximum 200 images to be downloaded
total_images = 200

# Existing downloaded image files
exiting_num_images = 1400

print("{0} - Try to download {1} images for '{2}'".format(
        show_time(), total_images, search_target))
print("---------")

num_images = complete_download_task(py_pexels, search_target, total_images, 
                       ignore_num_images = exiting_num_images)
print("{0} - Downloaded {1} images for '{2}'".format(
        show_time(), num_images, search_target))

