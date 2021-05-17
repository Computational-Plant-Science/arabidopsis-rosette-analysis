# USAGE
# python3 superpixel.py --image 01.jpg

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

from skimage import data, segmentation, measure, color, img_as_float

import matplotlib.pyplot as plt
import argparse

import numpy as np
import cv2


import imutils

'''
def mean_image(image,label):

    im_rp = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    
    segments_1d = np.reshape(label, -1)    
    
    unique_label = np.unique(segments_1d)
    
    uu = np.zeros(im_rp.shape)
    
    mask_bk = np.zeros_like(image)
    
    for i in unique_label:

        loc = np.where(segments_1d == i)[0]
        print(loc)
        
        (x, y) = np.unravel_index(int(np.mean(loc)),(image.shape[0], image.shape[1]))
        
        print(x, y)
        
        #(x, y) = np.mean(np.nonzero(segments_1d == i),axis=1)
        
        region_label = cv2.putText(image, "#{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        mm = np.mean(im_rp[loc,:], axis = 0)
        uu[loc,:] = mm
    
    cv2.imwrite('region_label.png', region_label)
    
    output = np.reshape(uu,[image.shape[0],image.shape[1],image.shape[2]]).astype('uint8')
    
    #output = mark_boundaries(image, output)
    
    return output



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())


img = cv2.imread(args["image"])

#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

label = segmentation.slic(img, compactness = 10, n_segments = 10)

output = mean_image(img,label)

cv2.imwrite('seg.png', output)


'''





from skimage.segmentation import slic
from scipy.spatial import Delaunay
from skimage.segmentation import mark_boundaries
from matplotlib.lines import Line2D
from skimage import measure, color, img_as_float

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
img = img_as_float(io.imread(args["image"]))

# SLIC
segments = slic(img, n_segments=100, compactness=20)
segments_ids = np.unique(segments)


# loop over the unique segment values, Accessing Individual Superpixel Segmentations
for (i, segVal) in enumerate(np.unique(segments)):
    # construct a mask for the segment
 
    print("inspecting segment : {0}\n".format(i))
    
    mask = np.zeros(img.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    
    # show the masked region
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Applied", cv2.bitwise_and(img, img, mask = mask))
    #cv2.waitKey(0)



# centers
centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.imshow(mark_boundaries(img, segments))
plt.scatter(centers[:,1],centers[:,0], c='y')


for i in range(bneighbors.shape[1]):
    y0,x0 = centers[bneighbors[0,i]]
    y1,x1 = centers[bneighbors[1,i]]

    l = Line2D([x0,x1],[y0,y1], alpha=0.5)
    ax.add_line(l)

plt.show()




