# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:09:43 2021

@author: Admin
"""

import argparse
import sys
from histogram_image_ import* 
from build_image_file_list import*

directory = './subImage/Bearing/'
height_csv = './bbox/height.csv'
width_csv = './bbox/width.csv'
dictionary_det = './bbox/dict.csv'
bins = 5
image_file_paths, image_names = buildImageFileList(directory)
df = img_height_width_csv(image_file_paths, dictionary_det)
plotHistogram(df, bins)
