# USAGE
# python3 colorful_regions.py --image example_01.jpg

# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2


def segment_colorfulness(image, mask):
    # split the image into its respective RGB components, then mask
    # each of the individual RGB channels so we can compute
    # statistics only for the masked region
    (B, G, R) = cv2.split(image.astype("float"))
    R = np.ma.masked_array(R, mask=mask)
    G = np.ma.masked_array(B, mask=mask)
    B = np.ma.masked_array(B, mask=mask)

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`,
    # then combine them
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)
    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,    help="path to input image")
ap.add_argument("-s", "--segments", type=int, default=100,    help="# of superpixels")
args = vars(ap.parse_args())

# load the image in OpenCV format so we can draw on it later, then
# allocate memory for the superpixel colorfulness visualization
orig = cv2.imread(args["image"])
vis = np.zeros(orig.shape[:2], dtype="float")

# load the image and apply SLIC superpixel segmentation to it via
# scikit-image
image = io.imread(args["image"])
segments = slic(img_as_float(image), n_segments=args["segments"],    slic_zero=True)

# loop over each of the unique superpixels
for v in np.unique(segments):
    # construct a mask for the segment so we can compute image statistics for *only* the masked region
    mask = np.ones(image.shape[:2])
    mask[segments == v] = 0

    # compute the superpixel colorfulness, then update the visualization array
    C = segment_colorfulness(orig, mask)
    vis[segments == v] = C

# scale the visualization image from an unrestricted floating point
# to unsigned 8-bit integer array so we can use it with OpenCV and
# display it to our screen
vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")

#define result path for labeled images
result_img_path = 'vis.png'
cv2.imwrite(result_img_path, vis)

# overlay the superpixel colorfulness visualization on the original image
alpha = 0.5
overlay = np.dstack([vis] * 3)
output = orig.copy()
res = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

result_img_path = 'overlay.png'
cv2.imwrite(result_img_path, res)

# show the output images
cv2.imshow("Input", orig)
cv2.imshow("Visualization", vis)
cv2.imshow("Output", output)
cv2.waitKey(0)



cv2.destroyAllWindows()




'''
from skimage import data, segmentation, measure, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

segments = segmentation.slic(image, n_segments=100, compactness=20)

#segments = segmentation.slic(image, slic_zero=True)

regions = measure.regionprops(segments)

colorfulness = np.zeros(image.shape[:2])

for region in regions:
    # Grab all the pixels from this region
    # Return an (N, 3) array
    coords = tuple(region.coords.T)
    values = image[coords]

    R, G, B = values.T

    # Calculate colorfulness
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    colorfulness[coords] = std_root + (0.3 * mean_root)

hsv = color.rgb2hsv(image)
hsv[..., 2] *= colorfulness / colorfulness.max()

plt.imshow(color.hsv2rgb(hsv))
plt.savefig('vis.png')
'''
