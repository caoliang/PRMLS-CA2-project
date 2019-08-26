# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:56:15 2019

@author: Cao Liang
"""

import json
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import re

class DownloadImagesInfo():
    EMPTY_CHAR_REGEX = re.compile(r'[\s\r\n]+')
    SPACE_CHAR_REGEX = re.compile(r'\s')
    AMP_CHAR_REGEX = re.compile(r'&\s*')
    REG_CHAR_REGEX = re.compile('®|™')

    def __init__(self):
        self.imgs_info = {}

    ## Read objects from json file
    def read_objs(self, json_file):
        with open(json_file, mode="r", encoding="UTF-8") as json_file:
            data_objs = json.loads(json_file.read())

        return data_objs

    ## Write objects to json file
    def write_objs(self, info_obj, json_file):
        with open(json_file, mode="w", encoding="UTF-8") as json_file:
            json.dump(info_obj, json_file, indent=4)

    def prepare_course_data(self):
        info_data = self.read_objs("executive-courses.json")
        links_data = self.read_objs("all_link.json")

        for course_info in info_data:
            course_name_link = course_info['name'].lower().strip()
            # nicf- or nicf– (special character E2 80 93) so remove 5 chars
            if course_name_link.startswith('nicf'):
                course_name_link = course_name_link[5:].strip()
            # (isc)2 or (isc)² so remove 6 chars
            if course_name_link.startswith('(isc)'):
                course_name_link = course_name_link[6:].strip()
            if course_name_link.startswith('itil®'):
                course_name_link = course_name_link[5:].strip()
            #if course_name_link.startswith('pmp®'):
            #    course_name_link = course_name_link[4:].strip()
            if course_name_link.endswith('(sf)'):
                course_name_link = course_name_link[:-4].strip()
            if course_name_link.endswith('- exams supported by citrep+'):
                course_name_link = course_name_link[:-28].strip()
            if course_name_link.endswith('(cissp exam only)'):
                course_name_link = course_name_link[:-17].strip()
            if course_name_link.endswith('(ccsp exam only)'):
                course_name_link = course_name_link[:-16].strip()
            if course_name_link.endswith('and seo'):
                course_name_link = (course_name_link[:-7] + 'seo').strip()
            # Add one more space to construct correct url
            if course_name_link.endswith('(pmi-acp® exam only)'):
                course_name_link = (course_name_link[:-20] + ' (pmi-acp-exam-only)').strip()
            if course_name_link.endswith('(pmp® exam only)'):
                course_name_link = (course_name_link[:-17] + '  (pmp exam only)').strip()
            if course_name_link.endswith('e-government leadership'):
                course_name_link = 'e-government'
            if course_name_link.endswith(' - capstone & internship'):
                course_name_link = course_name_link[:-24] + '-capstone-internship'
            if course_name_link == 'nus-iss certificate in digital solutions development – design':
                course_name_link = 'nus-graduate-diploma-in-systems-analysis-(sf)'
            if course_name_link == 'nus-iss certificate in digital solutions development – foundations':
                course_name_link = 'nus-iss-certificate-in-digital-solutions-development'
            if course_name_link == 'nus-iss certificate in digital solutions development – web applications':
                course_name_link = 'nus-iss-certificate-in-digital-solutions-development-web-applications'
            if course_name_link == 'nus-iss stackable certificate programmes in digital solutions development':
                course_name_link = 'nus-iss-certificate-in-digital-solutions-development'
            # use code for course url
            if 'secure software development lifecycle for agile' in course_name_link:
                course_name_link = "ssdla-sf"

            course_name_link = ExecutiveCourseInfo.REG_CHAR_REGEX.sub('', course_name_link)
            course_name_link = ExecutiveCourseInfo.AMP_CHAR_REGEX.sub('', course_name_link)
            course_name_link = ExecutiveCourseInfo.SPACE_CHAR_REGEX.sub('-', course_name_link)
            print(("course name link: '{}'").format(course_name_link))
            course_url = None
            for link_url in links_data:
                if link_url.find(course_name_link) >= 0:
                    course_url = link_url
                    break

            if course_url is None:
                print("Cannot find link for course: {}".format(course_info["name"]))
                break
            else:
                print("Course: {}, link: {}".format(course_info["name"], course_url))

            course_info["course_link"] = course_url

            # Add course details info from web page
            course_content = self.read_url_content(course_url)
            self.parse_content(course_content, course_info)

        self.write_objs(info_data, "ExecutiveCourses.json")

    ## Save sample html for testing
    def save_content(self, content, local_file_path):
        with open(local_file_path, mode="w", encoding="UTF-8") as test_file:
            test_file.write(content)

    ## Read URL content
    def read_url_content(self, url_link):
        url_content = requests.get(url_link)
        return url_content.text

    ## Read local file content
    def read_file_content(self, local_file_path):
        with open(local_file_path, mode="r", encoding="UTF-8") as test_file:
            data = test_file.read()
        return data

    ## Clean up html content data to remove redundant white space
    def clean_html_data(self, html_data):
        return ExecutiveCourseInfo.EMPTY_CHAR_REGEX.sub(" ", html_data.strip())

    ## Parse html with beautiful soup
    def parse_content(self, content, content_info):
        soup_obj = BeautifulSoup(content, "html.parser")
        #self.show_full_conent(soup_obj)

        # Read fullname, Reference No, Duration, Course Time, Enquiry
        # and description (in string list format)
        self.parse_overview(soup_obj, content_info)

        # Read takeaway, whoattend, whatcovered;
        # All are in string list format
        self.parse_details(soup_obj, content_info)

        return content_info

    def show_full_conent(self, soup_data, save_parsed_file = "parsed.html"):
        parsed_content = soup_data.prettify()
        #print(parsed_content)
        self.save_content(parsed_content, save_parsed_file)

    def parse_overview(self, soup_data, details_info):
        overview_info = {}
        details_info["overview"] = overview_info

        full_name = soup_data.select_one("h1#hdrTitle").get_text("").strip()
        overview_info["full_name"] = full_name
        print("full_name: {}".format(full_name))
        overview_div = soup_data.select_one("div#overview")
        row_list = overview_div.select("table tbody tr")
        for row in row_list:
            th_part = self.clean_html_data(row.select_one("th").get_text(""))
            td_part = self.clean_html_data(row.select_one("td").get_text(""))
            overview_info[th_part] = td_part
            print("th: '{}', td: '{}'".format(th_part, td_part))

        desc_list = []
        for p_part in overview_div.select("p"):
            if type(p_part) == NavigableString:
                desc = self.clean_html_data(p_part)
                #print("desc string = {}".format(desc))
            else:
                desc = self.clean_html_data(p_part.get_text(""))
                #print("desc = {}".format(desc))

            if desc != "":
                desc_list.append(desc)

        overview_info["description"] = desc_list
        for desc in desc_list:
            print("desc item: {}".format(desc))

    def parse_details(self, soup_data, details_info):
        details_info["takeaway"] = []
        self.parse_details_list(soup_data, "takeaway", "div#tab1 ul:nth-of-type(1) li", details_info["takeaway"])
        if len(details_info["takeaway"]) == 0:
            self.parse_details_list(soup_data, "takeaway", "div#tab1 li", details_info["takeaway"])

        details_info["whoattend"] = []
        info_list = soup_data.select("div#tab2")[0].get_text().splitlines()
        for info_item in info_list:
            info_text = self.clean_html_data(info_item)
            info_text_check = info_text.lower()
            if info_text_check == 'pre-requisites' or info_text_check == 'prerequisites' \
                or info_text_check == 'pre-requsites' \
                or info_text_check == 'requirements' or info_text_check == 'what to bring' \
                or info_text_check == 'important notes':
                break

            if info_text_check != "" and info_text_check != 'who should attend'\
                    and not info_text_check.startswith("this is") \
                    and not info_text_check.startswith('this course'):
                details_info["whoattend"].append(info_text)
                print("Found whoattend item: {}".format(info_text))

        details_info["whatcovered"] = []
        self.parse_details_list(soup_data, "whatcovered", "div#tab3 ul:nth-of-type(1) li", details_info["whatcovered"])
        if len(details_info["whatcovered"]) == 0:
            self.parse_details_list(soup_data, "whatcovered", "div#tab3 li", details_info["whatcovered"])

        if len(details_info["whatcovered"]) == 0:
            info_list = soup_data.select("div#tab3")[0].get_text().splitlines()
            for info_item in info_list:
                info_text = self.clean_html_data(info_item)
                if info_text != "" and info_text.lower() != 'what will be covered':
                    details_info["whatcovered"].append(info_text)
                    print("Found whatcovered item: {}".format(info_text))

    def parse_details_list(self, soup_data, item_name, item_css, list_info):
        item_list = soup_data.select(item_css)
        for item in item_list:
            item_desc = self.clean_html_data(item.getText(""))
            print("{} item: {}".format(item_name, item_desc))
            if item_desc != "":
                list_info.append(item_desc)

    def create_test_file(self, test_url, save_file = "test.html"):
        test_content = self.read_url_content(test_url)
        self.save_content(test_content, save_file)

    def read_test_file(self, save_parsed_file="parsed.html", save_file="test.html"):
        test_content_info = {}
        test_content = self.read_file_content(save_file)
        test_content_parsed = BeautifulSoup(test_content, "html.parser")
        self.show_full_conent(test_content_parsed, save_parsed_file)
        self.parse_content(test_content, test_content_info)

if __name__ == "__main__":
    print('Download images')
    imagesInfo = DownloadImagesInfo()
    # test_url = "https://www.iss.nus.edu.sg/executive-education/course/detail/nus-iss-certificate-in-digital-solutions-development-foundations-(sf)"
    # courseInfo.create_test_file(test_url)
    # courseInfo.read_test_file(save_parsed_file='parsed21.html')

    # for idx in range(1, 10):
    #     courseInfo.read_test_file("parsed{}.html".format(idx))

    #courseInfo.prepare_course_data()


