# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:56:15 2019

@author: Cao Liang
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver import ActionChains
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.error import HTTPError
from datetime import datetime
import base64
import time
import sys
import traceback
import json
import requests
import re
import urllib
import os


def show_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def get_hours_spent(start_time):
    return int((time.time() - start_time) / 3600)

def show_time_spent(start_time):
    time_used = time.time() - start_time
    return "{}:{}:{}".format(int(time_used / 3600), int(time_used % 3600 / 60), int(time_used % 60))

def get_web_file_folder(search_words):
    save_file_folder = "public_{0}".format(search_words).replace(' ', '_')
    return save_file_folder

def write_log(log_path, msg):
    time_tag = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
    with open(log_path, 'a') as logger:
        logger.write("{0} - {1}".format(time_tag, msg))
        logger.write("\n")
    
    print("{0} - {1}".format(time_tag, msg))    

def write_log_error(log_path, msg):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    time_tag = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

    with open(log_path, 'a') as logger:
        logger.write("{0} - {1}".format(time_tag, msg))
        logger.write("\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=logger)

    print("{0} - {1}".format(time_tag, msg))
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


def log(msg):
    write_log('searchimg.log', msg)

def log_error(msg):
    write_log_error('searchimg.log', msg)


class DownloadFromSnappygoat():
    IMG_ID_REGEX = re.compile(r'[0-9a-z]+', re.IGNORECASE)
    IMG_LINK_REGEX = re.compile(r'^/t/([0-9a-z]+)', re.IGNORECASE)
    IMG_FULL_LINK_REGEX = re.compile(r'^https://snappygoat.com/t/([0-9a-z]+)', 
                                     re.IGNORECASE)
    IMG_EXT_REGEX = re.compile(r'.+\.([a-z]+)', re.IGNORECASE)
    IMG_BIN_REGEX = re.compile((r'^data:image/([a-z]+);base64,(.+)'), 
                               re.IGNORECASE)
    
    def __init__(self, search_words, start_page=1, total_num_download=1200):
        # gecko driver path
        geckodriver_path = 'geckodriver/geckodriver.exe'
        # Firefox exe file path
        firefox_exe_path = r"C:\\Program Files\\Mozilla Firefox\\firefox.exe"
        firefox_binary = FirefoxBinary(firefox_exe_path)
        self.driver = webdriver.Firefox(executable_path=geckodriver_path,
                                        firefox_binary=firefox_binary)
        self.driver.implicitly_wait(30)
        self.verificationErrors = []
        self.accept_next_alert = True
        kw_url = urllib.parse.quote(search_words)
        self.base_url = "https://snappygoat.com"
        self.url_full = self.base_url + "/s/?q={0}".format(kw_url)
        self.img_folder = get_web_file_folder(search_words)
        self.id_list_file = os.path.join(self.img_folder, 'id_list.txt')
        self.image_id_map = {}
        self.load_image_id_map()
        self.start_page_num = start_page
        self.total_images_to_download = total_num_download
    
    def load_image_id_map(self):
       if not os.path.exists(self.id_list_file):
           return
       
       lines = None
       with open(self.id_list_file, 'r') as img_id_list_file:
           lines = img_id_list_file.readlines()
       
       if lines is not None:
           for line in lines:
               content = line.strip()
               if content != '':
                   self.image_id_map[content] = ''
    
    def save_image_id_map(self):
        if len(self.image_id_map) == 0:
            return
        
        with open(self.id_list_file, 'w') as img_id_list_file:
           for id_key in self.image_id_map.keys():
               img_id_list_file.write(id_key)
               img_id_list_file.write('\n')
    
    def download_search_page(self):
        driver = self.driver
        driver.get(self.url_full)
        # loop through all image links
        self.download_images_from_search_page()
    
    def download_images_from_search_page(self):
        driver = self.driver
        
        has_next_page = True
        page_num = 0
        # Each page has 100 images 
        total_page = int(self.total_images_to_download / 100)
        log('Total page: {0}'.format(total_page))
        # Total number of images downloaded
        total_img = 0
        
        while has_next_page and page_num <= total_page:
            img_fields =  driver.find_elements_by_xpath(
                    '//a[@class="imres"]/img[@id]')
            if img_fields is None:
                log('No image field is found, so exit')
                break
            
            page_num += 1
            total_img_in_page = 0
            
            if self.start_page_num > 0 and page_num < self.start_page_num:
                log('Scroll to start page: {0} from current page: {1}'.format(
                        self.start_page_num, page_num))
                # Determin next page by scrolling to next page
                has_next_page = self.scroll_to_next_page()
                continue
            
            log('Start to downloaded images at page: {0}'.format(page_num))
            
            for img_tag in img_fields:
                try:
                    dl_result = self.download_image(img_tag)
                    log('Download img result: {0}'.format(dl_result))
                    if dl_result:
                        total_img += 1
                        total_img_in_page += 1
                except:
                    log_error('Download img with error')
                            
            log('Downloaded images (current page:{0}, total: {1})'.format(
                    total_img_in_page, total_img))
            
            if page_num <= total_page:
                # Determin next page by scrolling to next page
                has_next_page = self.scroll_to_next_page()
        
        log('Total downloaded images: {0}'.format(total_img))
        # Save image id list
        self.save_image_id_map()
    
    def scroll_to_next_page(self):
        driver = self.driver
        # page load time in seconds
        wait_page_load_time = 5

        scroll_page_ok = False

        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        log('last_height: {0}'.format(last_height))

        # Scroll down to bottom
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.execute_script("window.scrollTo(0, {0})".format(last_height))
    
        # Wait to load page
        time.sleep(wait_page_load_time)
        
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        log('new_height: {0}'.format(new_height))

        diff_height = new_height - last_height
        log('diff_height: {0}'.format(diff_height))

        if diff_height > 0:
            scroll_page_ok = True
        
        log('scroll_page_ok: {0}'.format(scroll_page_ok))
        
        return scroll_page_ok
    
    def download_image(self, img_tag):
        #log('img_tag method: {}', dir(img_tag))
        img_id = img_tag.get_attribute("id")
        if img_id is None:
            log('Image ID Not found'.format(img_id))
            return False
        
        log('img_id: "{0}"'.format(img_id))
        if img_id in self.image_id_map:
            log('Aleady downloaded img_id: "{0}"'.format(img_id))
            return False    

        img_alt = img_tag.get_attribute("alt")
        if img_alt is None:
            log('Image: {0} Alt Not found'.format(img_id))
            return False
        
        alt_match = re.search(DownloadFromSnappygoat.IMG_EXT_REGEX, img_alt)
        if alt_match is None:
            log('Image: {0} extension Not found'.format(img_id))
            return False
        
        img_ext = alt_match.group(1)
        img_name = '{0}.{1}'.format(img_id, img_ext)
        
        img_src = img_tag.get_attribute("src")
        if img_src is None or img_src.strip() == '':
            log('Image: {0} src field Not found'.format(img_id))
            return False
        
        src_match = re.search(DownloadFromSnappygoat.IMG_LINK_REGEX, img_src)
        if src_match:
            log('Image: {0} link found'.format(img_id))
            img_link = self.base_url + img_src
            
            # Download every images
            dl_result = self.save_image_in_url(self.img_folder, 
                                               img_name, img_link)
            log('Downloading {0} result: {1}'.format(img_name, 
                  dl_result))
            if dl_result:
                self.image_id_map[img_id] = ''
            
            return dl_result
        
        full_link_match = re.search(DownloadFromSnappygoat.IMG_FULL_LINK_REGEX, 
                                    img_src)
        if full_link_match:
            log('Image: {0} full link found'.format(img_id))
            img_link = img_src
            
            # Download every images
            dl_result = self.save_image_in_url(self.img_folder, 
                                               img_name, img_link)
            log('Downloading {0} with full link result: {1}'.format(img_name, 
                  dl_result))
            if dl_result:
                self.image_id_map[img_id] = ''
            
            return dl_result
                
        img_bin_match = re.search(DownloadFromSnappygoat.IMG_BIN_REGEX, 
                                  img_src)
        if img_bin_match:
            img_ext = img_bin_match.group(1)
            img_name = '{0}.{1}'.format(img_id, img_ext)
            
            img_bytes = base64.b64decode(img_bin_match.group(2))
            save_result = self.save_image_in_bytes(self.img_folder,
                                                   img_name, img_bytes)
            log('Saving {0} result: {1}'.format(img_name, 
                                                  save_result))
            if save_result:
                self.image_id_map[img_id] = ''
            
            return save_result
        
        log('Cannot found img {0} src link or bytes for {1}'.format(img_id, 
            img_src))
        return False
                
    def save_image_in_url(self, folder_name, file_name, file_url):
        file_full_path = os.path.join(folder_name, file_name)
        
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                    
            url_resp = requests.get(file_url, allow_redirects=True)
            
            with open(file_full_path, "wb") as img_file:
                img_file.write(url_resp.content)
            
            return True
        except:
            log_error('Download {0} error'.format(file_url))
            
        return False
    
    def save_image_in_bytes(self, folder_name, file_name, file_bytes):
        file_full_path = os.path.join(folder_name, file_name)
        
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                    
            with open(file_full_path, "wb") as img_file:
                img_file.write(file_bytes)
            
            return True
        except:
            log_error('Save file {0} error'.format(file_name))
            
        return False

    def shut_down(self):
        self.driver.quit()

if __name__ == "__main__":
    # target images to search
    search_content = "cat"

    # Begining page number (default is 1 from beginning)
    start_page_number = 1

    time_start_point = time.time()
    log("Start to download images for '{0}' from page: {1}".format(
            search_content, start_page_number))
    
    try:
        downloader = DownloadFromSnappygoat(search_content, 
                                        start_page=start_page_number)
        # Show search page
        downloader.download_search_page()
    except:
        log_error('Error to download images')
    finally:
        # Shutdown browser
        downloader.shut_down()
        
    # Complete to download images
    log('Complete to download images in {0}'.format(
            show_time_spent(time_start_point)))
    

