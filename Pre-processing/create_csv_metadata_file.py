# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:23:16 2020

@author: Eric Bianchi
"""
import sys
import os

from build_image_file_list import buildImageFileList
from write_to_csv import arrayToCSV
from xml_to_sub_image import xml_to_sub_image

def makeInputSheet(folderName, fields): 
    imageFilePaths, image_names = buildImageFileList(folderName)
    brokenList=folderName.split('/')
    objectType = brokenList[len(brokenList)-2]
    destination = folderName + objectType+'.csv'
    arrayToCSV(image_names, fields, destination)

folderName = './subImage/Bearing/' 
# here you can add whatever to the list. 

#=============================================================================
# BEARING:
fields = ['Image Name', 'Bearing Class', '(1000) Corrosion', 
          '(1020) Connection', '(2210) Movement', '(2220) Alignment', 
          '(2230) Bulging, Splitting, or Tearing', '(2240) Loss of Bearing Area', 
          '(7000) Damage']

# GUSSET PLATE CONNECTION:
"""
fields = ['Image Name', '(1000) Corrosion', '(1010)' Cracking,
          '(1020) Connection', '(1900) Distortion', '(7000) Damage']
"""

# COVERPLATE TERMINATION:
"""
fields = ['Image Name', 'Overall']
"""
    
# OUT OF PLANE STIFFENER:
"""
fields = ['Image Name', 'Overall']
"""
#=============================================================================

# xml_to_sub_image(xml_dir, csv_file, src_image_dir, dest_subimage_dir)
xml_to_sub_image('./bbox/xml/', './bbox/details.csv', './Images/', './subImage/')

# makeInputSheet(name of object folder, corresponding fields to include)
makeInputSheet(folderName, fields)