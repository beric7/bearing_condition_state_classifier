# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

sys.path.insert(0, 'E://Python/general_utils/')

from image_utils import random_sort_images

# blackAndWhite(source_image_folder, destination):
source_image = './1/'
destination_image_test = './Test/1/'
destination_image_train = './Train/1/'
percentage = 0.1

# random_sort_images(source_image_folder, destination, seed=10, percentage=0.1)
random_sort_images(source_image, destination_image_test, 
                   destination_image_train, percentage=percentage)
