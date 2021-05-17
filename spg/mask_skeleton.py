'''
Name: mask_skeleton.py

Version: 1.0

Summary: skeletonize binary mask to 1 pixel wide representations. 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2019-09-29

USAGE:

python3 mask_skeleton.py -p /home/suxing/plant-image-analysis/mask/  -ft jpg


'''

# import the necessary packages

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import morphology
#from skimage.morphology import medial_axis
from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
import os
import argparse

import glob

    
from skan import Skeleton, summarize, draw



        


def load_image(image_path):
    
    image = cv2.imread(image_path)
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    abs_path = os.path.abspath(image_path)
    
    filename, file_extension = os.path.splitext(image_path)
    
    
    return thresh, filename, abs_path
    
    
def skeleton_bw(mask):

    # Convert mask to boolean image, rather than 0 and 255 for skimage to use it
    
    skeleton = morphology.skeletonize(mask.astype(bool))
        
    skeleton = skeleton.astype(np.uint8) * 255

  
    return skeleton
    



if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    args = vars(ap.parse_args())
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    print(imgList)
    
    
    
    abs_path = os.path.abspath(imgList[0])
    
    
    
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # make the folder to store the results
    mkpath = os.path.dirname(file_path) +'/' + base_name
    mkdir(mkpath)
    save_path = mkpath + '/'
    print ("results_folder: " + save_path)
    
    
   
    source_image = cv2.cvtColor(cv2.imread(imgList[0]), cv2.COLOR_BGR2RGB)
    
    
    
    (image_mask, filename, abs_path) = load_image(imgList[0])
    
    image_skeleton = skeleton_bw(image_mask)
    
    result_file = (save_path + base_name + '_skeleton.' + ext)
    print(result_file)
    cv2.imwrite(result_file, img_as_ubyte(image_skeleton))


         
    fig = plt.plot()

    branch_data = summarize(Skeleton(image_skeleton))
    
    print(branch_data.head())
    
    fig = plt.plot()
    
    branch_data.hist(column='branch-distance', by='branch-type', bins = 100)
    
    result_file = (save_path + base_name + '_hist.' + ext)
    
    plt.savefig(result_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    
    
    
    fig = plt.plot()
    
    draw.overlay_euclidean_skeleton_2d(source_image, branch_data, skeleton_color_source = 'branch-distance', skeleton_colormap = 'hsv')
    
    result_file = (save_path + base_name + '_overlay.' + ext)
    
    plt.savefig(result_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    
    print(filename, abs_path)

    

    
    
    

    

    

    
