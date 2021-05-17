'''
Name: color_compare.py

Version: 1.0

Summary: Compare color difference in deltaE_cie76 space and estimate temperature difference. 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2019-09-29

USAGE:

python3 color_compare.py -p /home/suxingliu/plant-image-analysis/  -ft jpg


'''

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import argparse

import glob


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
    
def get_colors(image, number_of_colors, show_chart):
    
    #modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors


def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            print("Color difference value is : {0} \n".format(str(diff)))
            select_image = True
    
    return select_image

def show_selected_images(images, color, threshold, colors_to_match):
    
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i],
                                        color,
                                        threshold,
                                        colors_to_match)
        if (selected):
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            index += 1


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--image', required = True, help = 'Path to image file')
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")
    #ap.add_argument('-s', '--color-space', type =str, default ='lab', help='Color space to use: BGR (default), HSV, Lab, YCrCb (YCC)')
    #ap.add_argument('-c', '--channels', type = str, default='1', help='Channel indices to use for clustering, where 0 is the first channel,'  + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" '  + 'selects channels B and R. (default "all")')
    #ap.add_argument('-n', '--num-clusters', type = int, default = 2,  help = 'Number of clusters for K-means clustering (default 3, min 2).')
    args = vars(ap.parse_args())
    
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    
    #args_colorspace = args['color_space']
    #args_channels = args['channels']
    #args_num_clusters = args['num_clusters']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    #print(imgList)
    
    #rgb_colors = get_colors(get_image(imgList[0]), 8, True)
    
    #print(rgb_colors)
    
    
    IMAGE_DIRECTORY = image_file_path
    
    COLORS = {
    'GREEN': [0, 128, 0],
    'BLUE': [0, 0, 128],
    'YELLOW': [255, 255, 0]
    }
    
    images = []

    for image_file in imgList:
        images.append(get_image(image_file))
    
    '''
    plt.figure(figsize=(20, 10))
    
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
    
    plt.show()
    '''
    
    # Search for GREEN
    plt.figure(figsize = (20, 10))
    #show_selected_images(images, COLORS['GREEN'], 60, 5)
    #show_selected_images(images, COLORS['BLUE'], 60, 5)
    show_selected_images(images, COLORS['YELLOW'], 60, 5)
    
    plt.show()

    #print((imgList))
    
    #current_img = imgList[0]
    
    #(thresh, trait_img) = segmentation(current_img)
    
    '''
    # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(segmentation, imgList)
        pool.terminate()
    '''

    

    

    

    
