# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

sys.path.insert(0, 'E://Python/general_utils/')

from image_utils import rescale

# rescale(source_image_folder, destination, dimension):
dimension = 300
source = 'Train/1/'
destination = 'Train_300x300/1/'
rescale(source, destination, dimension)
