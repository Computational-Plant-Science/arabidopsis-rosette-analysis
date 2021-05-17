'''
Name: color_segmentation.py

Version: 1.0

Summary:  Extract plant traits (leaf area, width, height, ) by paralell processing 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2018-09-29

USAGE:

python3 color_kmeans_vis.py -p /home/suxingliu/plant-image-analysis/sample_test/ -i 01.jpg -m 01_seg.jpg -c 5


'''

#!/usr/bin/python

# import the necessary packages
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
import matplotlib.image as mpimg
import pylab as P
import os

def mkdir(path):
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?
    # True
    # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        print (path+'folder constructed!')
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        print (path+'path exists!')
        return False
        

def color_quantization(image, mask):
    
    #grab image width and height
    (h, w) = image.shape[:2]
    
    #change the color storage order
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #apply the mask to get the segmentation of plant
    masked_image = cv2.bitwise_and(image, image, mask = mask)
       
    # reshape the image to be a list of pixels
    pixels = masked_image.reshape((masked_image.shape[0] * masked_image.shape[1], 3))
        
    ############################################################
    #Clustering process
    ###############################################################
    # cluster the pixel intensities
    clt = MiniBatchKMeans(n_clusters = args["clusters"])
    #clt = KMeans(n_clusters = args["clusters"])
    clt.fit(pixels)

    #assign labels to each cluster 
    labels = clt.fit_predict(pixels)

    #obtain the quantized clusters using each label
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image_rec = pixels.reshape((h, w, 3))
    
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_RGB2BGR)
    image_rec = cv2.cvtColor(image_rec, cv2.COLOR_RGB2BGR)
    
    # display the images and wait for a keypress
    #cv2.imshow("image", np.hstack([image_rec, quant]))
    #cv2.waitKey(0)
    
    #define result path for labeled images
    result_img_path = save_path + 'cluster_out.png'
    
    # save color_quantization results
    cv2.imwrite(result_img_path,quant)

    # build a histogram of clusters and then create a figure representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)

    # remove the background color cluster
    if (args["mask"] == "None"):
        clt.cluster_centers_ = clt.cluster_centers_[1: len(clt.cluster_centers_)]
    else:
        clt.cluster_centers_ = clt.cluster_centers_[1: len(clt.cluster_centers_)]

    #build a histogram of clusters using center lables
    numLabels = utils.plot_centroid_histogram(save_path,clt)

    #create a figure representing the distribution of each color
    bar = utils.plot_colors(hist, clt.cluster_centers_)

    #save a figure of color bar 
    utils.plot_color_bar(save_path, bar)


if __name__ == '__main__':
        
        
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Current directory for image files.")
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-m", "--mask", required = True, help = "Path to the mask image", default = "None")
    ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
    args = vars(ap.parse_args())

    # setting path for results storage 
    current_path = args["path"]
    filename = args["image"]
    image_path = current_path + filename

    # construct result folder
    mkpath = current_path + str(filename[0:-4])
    mkdir(mkpath)
    
    global save_path
    save_path = mkpath + '/'
    print ("results_folder: " + save_path)
    
    # load the image
    image = cv2.imread(image_path)

    # set mask path
    mask_path = current_path + args["mask"]
    
    # load mask image as grayscale
    im_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    #extract the binary mask
    (thresh, mask) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    color_quantization(image,mask)
    
    '''
    #save mask image
    if (args["mask"] == "None"):
        
        #read mask image as gray scale
        im_gray = cv2.imread(fig_path_save, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #fill samll holes and area along edge of image
        from skimage.segmentation import clear_border
        
        # remove artifacts connected to image border
        cleared = im_bw.copy()
        im_bw_cleared = clear_border(cleared)
                
        #from skimage import morphology
        #im_bw_cleared = morphology.remove_small_objects(im_bw, 20000, connectivity=2)
        
        #remove small holes and objects 
        from scipy import ndimage as ndi
        label_objects, num_labels = ndi.label(im_bw_cleared)
        #print num_labels
        sizes = np.bincount(label_objects.ravel()) 
        mask_sizes = sizes > 500
        mask_sizes[0] = 0
        img_cleaned = mask_sizes[label_objects]
        
        #change output image type
        from skimage import img_as_ubyte
        img_cleaned = img_as_ubyte(img_cleaned)
        
        #save output mask image
        fig_name = (str(filename[0:-4]) + '_' +'mask.png')
        fig_path_mask = current_path + fig_name
        cv2.imwrite(fig_path_mask, img_cleaned)
        
        fig_path_mask = save_path + fig_name
        cv2.imwrite(fig_path_mask, img_cleaned)
    '''

