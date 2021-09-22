"""
@author: Eric Bianchi
"""
import os
import glob
from csv_to_sub_image import csv_to_sub_image
from xml_to_csv import xml_to_csv


def xml_to_sub_image(xml_dir, csv_file, src_image_dir, dest_subimage_dir):
    xml_to_csv(xml_dir, csv_file)
    print('converted xml to csv')
    csv_to_sub_image(csv_file, src_image_dir, dest_subimage_dir)
    print('converted csv to sub-image')
