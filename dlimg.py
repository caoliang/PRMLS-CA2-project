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
import time
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

class DownloadFromDigitalMedia():
    SEARCH_LINK_REGEX = re.compile((r'https://digitalmedia.fws.gov/'
                                   + r'digital/collection/natdiglib/'
                                   + r'id/([0-9]+)'), re.IGNORECASE)
    
    IMAGE_FORMAT_REGEX = re.compile((r'.*public\s+domain.*'), re.IGNORECASE)
    
    def __init__(self, search_words, start_page=1):
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
        self.base_url = "https://digitalmedia.fws.gov/"
        self.url_full = self.base_url + (("digital/collection/natdiglib/"
                         + "search/searchterm/{0}/field/all/mode/"
                         + "all/conn/all/order/nosort/ad/asc/page/{1}"
                         ).format(kw_url, start_page))
        self.img_folder = get_web_file_folder(search_words)
        
    def download_search_page(self):
        driver = self.driver
        driver.get(self.url_full)
        # loop through all image links
        self.check_search_links()
    
    def check_search_links(self):
        driver = self.driver
        
        has_next_page = True
        page_num = 0
        # Each page has 20 images, so 100 * 20 = 2000 images for 100 pages 
        total_page = 100
        
        while has_next_page and page_num <= total_page:
            next_page_control = None
            try:
                next_page_control = driver.find_element_by_xpath(
                    ('//div[@class="Paginator-paginationPageHolder "]/'
                    + 'ul/li/a[@title="Go to next page"]'))
            except Exception as err:
                print('Cannot find next page control: {0}'.format(err))
                has_next_page = False
            
            search_links = driver.find_elements_by_css_selector(
                    "a.SearchResult-container")
            if search_links is None:
                break
            
            page_num += 1
            
            for link in search_links:
                #print('link method: {}', dir(link))
                link_href = link.get_attribute("href")
                print('link href: "{0}"'.format(link_href))
                check_match = re.search(DownloadFromDigitalMedia.SEARCH_LINK_REGEX, 
                                       link_href)
                if check_match:
                    image_id = check_match.group(1)
                    print('Image: {0}, "{1}"'.format(image_id, link_href))
        
                    self.open_new_tab()
                
                    try:
                        # Download every images
                        self.download_image(image_id, link_href)
                    except Exception as err:
                        print('Downloading {0} error: {1}'.format(image_id, err))
                    finally:
                        # Close new tab
                        self.close_new_tab()
                    
            if not has_next_page:
                break
            
            next_page_control.click()
            
            search_links = driver.find_elements_by_css_selector(
                    "a.SearchResult-container")
            
    
    def save_image(self, folder_name, file_name, file_url):
        file_full_path = os.path.join(folder_name, file_name)
        
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                    
            url_resp = requests.get(file_url, allow_redirects=True)
            
            with open(file_full_path, "wb") as img_file:
                img_file.write(url_resp.content)
            
            return True
        except FileNotFoundError as err:
            print(err)   # something wrong with local path
        except HTTPError as err:
            print(err)
    
        return False
    
    def download_image(self, image_id, info_url):
        driver = self.driver
        
        # Open image link
        driver.get(info_url)

        # Image must be public domain        
        rights_field = driver.find_element_by_css_selector(
                "tr.field-rights td.field-value>span")
        if rights_field is None:
            print('NOT found right for image: {0}'.format(image_id))
            return False
        
        #print('rights_field: {0}, {1}', rights_field, dir(rights_field))
        check_img_rights = re.search(
                DownloadFromDigitalMedia.IMAGE_FORMAT_REGEX, 
                rights_field.text)
        if check_img_rights:
            print('Found public domain image: {0}'.format(image_id))
        else:
            print('NOT public image: {0}'.format(image_id))
            return False
        
        format_field = driver.find_element_by_css_selector(
                'tr.field-format td a')
        
        if format_field is None:
            print('NOT found format field for image: {0}'.format(image_id))
            return False
        
        # Get image format such as: jpg, png
        img_format = format_field.text.lower().strip()
        if img_format == '':
            print('NOT found format for image: {0}'.format(image_id))
            return False
        
        img_name = '{0}.{1}'.format(image_id, img_format)
        
        # Get image URL for downloading
        url_field = driver.find_element_by_css_selector(
                'div.ItemImage-itemImage>div>img')
        
        if url_field is None:
            print('NOT found URL field for image: {0}'.format(image_id))
            return False
            
        img_url = url_field.get_attribute('src')
        if img_url == '':
            print('NOT found url for image: {0}'.format(image_id))
            return False
        
        # Download image
        save_result = self.save_image(self.img_folder, img_name, img_url)
        print('Save image {0} result: {1}'.format(img_name, save_result))
       
        return save_result
        
    # Open new tab
    def open_new_tab(self):
        driver = self.driver
        
        # Open new tab if not exits
        if len(driver.window_handles) == 1:
            # Open a new window
            driver.execute_script("window.open('');")

        # Switch to the new window
        driver.switch_to.window(driver.window_handles[1])

    # Close new tab
    def close_new_tab(self, close_all = False):
        driver = self.driver
        
        # Only close new tab when all is completed
        if close_all:
            driver.close()
            
        # Switch back to the first tab with URL A
        driver.switch_to.window(driver.window_handles[0])
        
    def shut_down(self):
        self.driver.quit()
        
class DownloadImagesInfo():

    def __init__(self):
        self.images_info = {}

    ## Read objects from json file
    def read_objs(self, json_file):
        with open(json_file, mode="r", encoding="UTF-8") as json_file:
            data_objs = json.loads(json_file.read())

        return data_objs

    ## Read URL content
    def read_url_content(self, url_link):
        headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 5.1.1; SM-G928X Build/LMY47X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.83 Mobile Safari/537.36'}
        url_content = requests.get(url_link, headers=headers)
        return url_content.text

    ## Read local file content
    def read_file_content(self, local_file_path):
        with open(local_file_path, mode="r", encoding="UTF-8") as test_file:
            data = test_file.read()
        return data

    def show_full_conent(self, soup_data, save_parsed_file = "parsed.html"):
        parsed_content = soup_data.prettify()
        #print(parsed_content)
        self.save_content(parsed_content, save_parsed_file)

    def get_web_filename(self, search_words):
        save_file_name = "{0}.html".format(search_words)
        return save_file_name

    def get_web_file_path(self, search_words):
        kw_name = search_words.strip().replace(' ', '_')
        
        save_file_folder = get_web_file_folder(kw_name)
        save_file_name =self.get_web_filename(kw_name)
        save_file_path = os.path.join(save_file_folder, save_file_name)
        
        return save_file_path

    def read_web_file(self, search_words="fish", save_to_file=False):
        kw_url = urllib.parse.quote(search_words)
        url_full = ("https://digitalmedia.fws.gov/digital/collection/"+
                    "natdiglib/search/searchterm/{0}/field/all/mode/"+
                    "all/conn/all/order/nosort/ad/asc").format(kw_url)
        
        print("{0} - Read web page '{1}'".format(show_time(), url_full))
        file_content = self.read_url_content(url_full)
                
        if save_to_file:
            save_file_path = self.get_web_file_path(search_words)
            save_file_folder = get_web_file_folder(search_words)
            
            if not os.path.exists(save_file_folder):
                os.makedirs(save_file_folder)
                    
            self.save_content(file_content, save_file_path)
            print("{0} - Save web page to '{1}'".format(show_time(), 
                  save_file_path))
            
        return file_content

    def read_search_page(self, web_content=None, web_file="test.html"):
        if web_content is None:
            web_content = self.read_file_content(web_file)
        
        bs_content = BeautifulSoup(web_content, "html.parser")
        
        json_pattern = re.compile(r"JSON.parse\('(\{.*?\}\}\})'\);", 
            re.MULTILINE | re.DOTALL)
        json_content = bs_content.find("script", text=json_pattern)
        
        special_pattern = re.compile(r"\\[xX]", re.IGNORECASE)
        json_data = json_pattern.search(json_content.text).group(1)
        json_data = re.sub(special_pattern, "\\u00", json_data)
        
        internal_quote_pattern = re.compile(r'\\\\\\"')
        json_data = re.sub(internal_quote_pattern, "'", json_data)
        outer_quote_pattern = re.compile(r'\\"')
        json_data = re.sub(outer_quote_pattern, '"', json_data)
                
        if web_file is not None:
            self.save_content(json_data, web_file + ".json")
            
        print('Loaded json data from search page: {0}'.format(json_data))
        json_obj = json.loads(json_data)
        print('Loaded json obj from search page: {0}'.format(json_obj))
       

if __name__ == "__main__":
    # target images to search
    search_content = "freshwater fish"
    # start page number
    start_page_num = 7

    time_start_point = time.time()
    print("{0} - Start to download images for '{1}'".format(
            show_time(), search_content))
    
    downloader = DownloadFromDigitalMedia(search_content, 
                                          start_page=start_page_num)
    
    # Show search page
    downloader.download_search_page()
    
    # Shutdown browser
    downloader.shut_down()
    
    # Complete to download images
    print('{0} - Complete to download images in {1}'.format(
            show_time(), show_time_spent(time_start_point)))
    

