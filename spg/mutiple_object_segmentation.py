'''
Name: color_segmentation.py
Version: 1.0
Summary: K-means color clustering based segmentation. This is achieved 
         by converting the source image to a desired color space and 
         running K-means clustering on only the desired channels, 
         with the pixels being grouped into a desired number
    of clusters. 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2020-09-29
USAGE:

python3 mutiple_object.py -p /home/suxing/plant-image-analysis/test/ -ft JPG

'''

# import the necessary packages
import os
import glob
import argparse
from sklearn.cluster import KMeans

from skimage.feature import peak_local_max
from skimage.morphology import watershed, medial_axis
from skimage import img_as_float, img_as_ubyte, img_as_bool, img_as_int
from skimage import measure

from skimage.segmentation import clear_border

from scipy.spatial import distance as dist
from scipy import optimize
from scipy import ndimage

import numpy as np
import argparse
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

MBFACTOR = float(1<<20)



# generate foloder to store the output results
def mkdir(path):
    # import module
    import os

    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")

    # path exist?   # True  # False
    isExists=os.path.exists(path)

    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False


def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters):

    # Change image color space, if necessary.
    colorSpace = args_colorspace.lower()

    if colorSpace == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    elif colorSpace == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    else:
        colorSpace = 'bgr'  # set for file naming purposes

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:,:,channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)

    (width, height, n_channel) = image.shape

    #print("image shape: \n")
    #print(width, height, n_channel)


    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])


    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')

    #define number of cluster
    numClusters = max(2, args_num_clusters)

    # clustering method
    kmeans = KMeans(n_clusters = numClusters, n_init = 40, max_iter = 500).fit(reshaped)

    # get lables 
    pred_label = kmeans.labels_

    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)],key = lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

    ret, thresh = cv2.threshold(kmeansImage,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thresh_cleaned = clear_border(thresh)
    
    return thresh_cleaned

'''
def medial_axis_image(thresh):
    
    #convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)
    image_bw = img_as_bool((thresh_sk))
    
    image_medial_axis = medial_axis(image_bw)
    
    return image_medial_axis
'''

def comp_external_contour(orig, thresh, save_path):

    #find contours and get the external one
    #image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    index = 1

    for c in contours:

        #get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        if w>60 and h>60:

            offset_w = int(w*0.15)
            offset_h = int(h*0.15)
            # draw a green rectangle to visualize the bounding rect
            roi = orig[y-offset_h : y+h+offset_h, x-offset_w : x+w+offset_w]

            print("ROI {} detected ...".format(index))

            result_file = (save_path +  str(format(index, "02")) + '.' + ext)

            #print(result_file)

            cv2.imwrite(result_file, roi)

            trait_img = cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 0), 3)

            trait_img = cv2.putText(orig, "#{}".format(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), 10)

            index+= 1

    return trait_img

def segmentation(image_file):

    abs_path = os.path.abspath(image_file)

    filename, file_extension = os.path.splitext(image_file)

    file_size = os.path.getsize(image_file)/MBFACTOR

    print("Segmenting image : {0} \n".format(str(filename)))

    # load original image
    image = cv2.imread(image_file)

    img_height, img_width, img_channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # make the folder to store the results
    #current_path = abs_path + '/'
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # save folder construction
    mkpath = os.path.dirname(abs_path) +'/' + base_name
    mkdir(mkpath)
    save_path = mkpath + '/'

    print("results_folder: {0}\n".format(str(save_path)))  

    if (file_size > 5.0):
        print("It will take some time due to large file size {0} MB".format(str(int(file_size))))
    else:
        print("Segmenting plant object using automatic color clustering method... ")

    #make backup image
    orig = image.copy()

    #color clustering based plant object segmentation
    thresh = color_cluster_seg(orig, args_colorspace, args_channels, args_num_clusters)

    #find external contour and segment image into small ROI based on each plant
    trait_img = comp_external_contour(image.copy(),thresh, save_path)

    result_file = abs_path +  '_label.' + ext

    cv2.imwrite(result_file, trait_img)

    return thresh, trait_img




if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--image', required = True, help = 'Path to image file')
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")

    ap.add_argument('-s', '--color-space', type =str, default ='lab', help='Color space to use: BGR, HSV, Lab, YCrCb (YCC)')
    ap.add_argument('-c', '--channels', type = str, default='1', help='Channel indices to use for clustering, where 0 is the first channel,' 
                                                                       + ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" ' 
                                                                       + 'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type = int, default = 2,  help = 'Number of clusters for K-means clustering (default 3, min 2).')
    args = vars(ap.parse_args())



    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']

    args_colorspace = args['color_space']
    args_channels = args['channels']
    args_num_clusters = args['num_clusters']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    #print((imgList))

    #current_img = imgList[0]

    #(thresh, trait_img) = segmentation(current_img)

     # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))

    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(segmentation, imgList)
        pool.terminate()

    #color clustering based plant object segmentation
    #thresh = color_cluster_seg(orig, args_colorspace, args_channels, args_num_clusters)

    # save segmentation result
    #result_file = (save_path + filename + '_seg' + file_extension)
    #print(filename)
    #cv2.imwrite(result_file, thresh)


    #find external contour 
    #trait_img = comp_external_contour(image.copy(),thresh)
    #save segmentation result
    #result_file = (save_path + filename + '_excontour' + file_extension)
    #cv2.imwrite(result_file, trait_img)


    #accquire medial axis of segmentation mask
    #image_medial_axis = medial_axis_image(thresh)

    # save medial axis result
    #result_file = (save_path + filename + '_medial_axis' + file_extension)
    #cv2.imwrite(result_file, img_as_ubyte(image_medial_axis))

