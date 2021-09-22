# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:55:35 2020

@author: Eric Bianchi
"""

import argparse
import sys

def main(*args):
    
    from histogram_image import* 
    args = args[0]
    sys.path.insert(0, args.module_path)

    plotHistogram(args.heightCSV, args.width_csv, args.module_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-module_path', type=str,  required=True, help='path where python modules are located')
    parser.add_argument('-heightCSV', type=str,  required=True, help='path where height CSV file is to be saved')
    parser.add_argument('-widthCSV', type=str,  required=True, help='path where width CSV file is to be saved')
    args = parser.parse_args()
    main(args)