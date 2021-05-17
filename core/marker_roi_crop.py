'''
Name: circle_detection.py

Version: 1.0

Summary: Detect circle shape markers in image and cropp image based on marker location
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2021-03-09

USAGE:

time python3 marker_roi_crop.py -p ~/plant-image-analysis/test/ -ft png 

'''

# import necessary packages
import argparse
from os.path import join

import cv2
import numpy as np
import os
import glob
from pathlib import Path

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

from tabulate import tabulate
import openpyxl

from pathlib import Path

# generate foloder to store the output results
from popos import InputImage


TEMPLATE_PATH = "/opt/spg-topdown-traits/marker_template/template.png"


def mkdir(path):
    # import module
    import os

    # remove space at the beginning
    path = path.strip()
    # remove slash at the end
    path = path.rstrip("\\")

    # path exist?   # True  # False
    isExists = os.path.exists(path)

    # process
    if not isExists:
        # construct the path and folder
        # print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        # print path+' path exists!'
        return False


def circle_detect(options: InputImage):
    # load the image, clone it for output, and then convert it to grayscale
    img_ori = cv2.imread(options.path)
    img_rgb = img_ori.copy()

    # Convert it to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Store width and height of template in w and h
    template = cv2.imread(TEMPLATE_PATH, 0)
    w, h = template.shape[::-1]

    # Perform match operations. 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold 
    threshold = 0.8

    # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold)

    if len(loc):
        (y, x) = np.unravel_index(res.argmax(), res.shape)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)

        print(y, x)
        print(min_val, max_val, min_loc, max_loc)

        # Draw a rectangle around the matched region. 
        for pt in zip(*loc[::-1]):
            circle_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

            # save segmentation result
        # result_file = (save_path + base_name + '_circle.' + args['filetype'])
        # print(result_file)
        # cv2.imwrite(result_file, circle_overlay)

        crop_img = img_rgb[y + 150:y + 850, x - 700:x]
        # save segmentation result
        cv2.imwrite(join(options.output_directory, f"{options.stem}.cropped.png"), crop_img)

    return options.name, (x, y), crop_img


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True, help="image filetype")
    ap.add_argument("-o", "--output_directory", required=True, help="directory to write output files to")

    args = vars(ap.parse_args())

    # Setting path to image files
    file_path = args["path"]
    file_type = args['filetype']
    output_dir = args['output_directory']

    # Extract file type and path
    filetype = '*.' + file_type
    image_file_path = file_path + filetype

    # Accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    options = [InputImage(input_path=file, output_directory=output_dir) for file in imgList]

    # Read the template
    template = cv2.imread(TEMPLATE_PATH, 0)
    print(template)

    # Get number of images in the data folder
    n_images = len(imgList)

    # get cpu number for parallel processing
    agents = psutil.cpu_count()
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))

    # Create a pool of processes. By default, one is created for each CPU in the machine.
    with closing(Pool(processes=agents)) as pool:
        results = pool.map(circle_detect, options)
        pool.terminate()

    # Output sum table in command window
    print("Summary: {0} plant images were processed...\n".format(n_images))
    table = tabulate(results, headers=['image_file_name', 'marker coordinates'], tablefmt='orgtbl')
    print(table + "\n")
