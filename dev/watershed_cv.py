"""
Version: 1.5

Summary: compute the segmentaiton and label of cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 watershed_cv.py -p /home/suxing/plant-image-analysis/test/  -ft jpg


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")

"""


# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
 
import glob
import os,fnmatch

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

def mkdir(path):
    """Create result folder"""
 
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
        
        
def image_label(image_file):
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    #Parse image path  and create result image path
    path, filename = os.path.split(image_file)

    print("processing image : {0} \n".format(str(filename)))

    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    image = cv2.imread(image_file)

    #imgcolor = ~imgcolor

    #accquire image dimensions 
    height, width, channels = image.shape
     
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Thresh", thresh)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance = 20, labels=thresh)
    
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    #Map component labels to hue val
    label_hue = np.uint8(128*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set background label to black
    labeled_img[label_hue==0] = 0

    #define result path for labeled images
    result_img_path = save_path_label + str(filename[0:-4]) + '_label.jpg'

    # save results
    cv2.imwrite(result_img_path,labeled_img)


    count = 0
    # loop over the unique labels returned by the Watershed algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
     
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        
        #define result path for simplified segmentation result
        result_img_path = save_path_ac + str(filename[0:-4]) + str(label) + '_ac.jpg'
        
        cv2.imwrite(result_img_path,mask)
     
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
     
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > 80:
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            count+= 1

    print("[INFO] {} unique segments found".format(count))

    #define result path for simplified segmentation result
    result_img_path = save_path_ac + str(filename[0:-4]) + '_ac.jpg'

    #write out results
    cv2.imwrite(result_img_path,image)



if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")
    args = vars(ap.parse_args())

    global save_path_ac, save_path_label
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
     
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    # make the folder to store the results
    mkpath = file_path + str('active_component')
    mkdir(mkpath)
    save_path_ac = mkpath + '/'
    
    mkpath = file_path + str('lable')
    mkdir(mkpath)
    save_path_label = mkpath + '/'

    #print "results_folder: " + save_path_ac  
    
    # Run this with a pool of avaliable agents having a chunksize of 3 until finished
    # run image labeling fucntion to accquire segmentation for each cross section image
    agents = multiprocessing.cpu_count() - 2
    chunksize = 3
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(image_label, imgList, chunksize)
        pool.terminate()
        
    
    
    
    
    

    
