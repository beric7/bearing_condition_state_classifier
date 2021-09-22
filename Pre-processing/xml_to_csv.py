"""
Date Altered: 5/1/2019

Note: substantial portions of this code, expecially the actual XML to CSV conversion, are credit to Dat Tran
see his website here: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
and his GitHub here: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

@author: Eric Bianchi
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import cv2

#######################################################################################################################
def xml_to_csv(srcDirectory, destDirectory):
    # convert training xml data to a single .csv file
    print("converting xml training data . . .")
    trainCsvResults = xml_to_df(srcDirectory)
    trainCsvResults.to_csv(destDirectory, index=None)
    print("training xml to .csv conversion successful, saved result to " + destDirectory)

# end main

#######################################################################################################################
def xml_to_df(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        try:
            filename.split('.')[1]+'.jpeg'
        except:
            filename = filename + '.jpeg'
                
        if (filename.split('.')[1] == 'png'):
            file = filename.split('.')[0]+'.jpeg'
            img = cv2.imread(path+file)
            height = int(img.shape[0])
            width = int(img.shape[1])
            for member in root.findall('object'):
                
                value = (file, width, height, member[0].text,
                          int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
                xml_list.append(value)
        else:
            for member in root.findall('object'):
    
                file = filename
                value = (file, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                          int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
                xml_list.append(value)
        # end for
    # end for

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
# end function

#######################################################################################################################
