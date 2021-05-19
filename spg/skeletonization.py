import numpy as np
from skimage import img_as_float, img_as_bool, morphology


def skeleton_bw(thresh):
    # Convert mask to boolean image, rather than 0 and 255 for skimage to use it

    # convert an image from OpenCV to skimage
    thresh_sk = img_as_float(thresh)

    image_bw = img_as_bool((thresh_sk))

    skeleton = morphology.skeletonize(image_bw)

    skeleton_img = skeleton.astype(np.uint8) * 255

    return skeleton_img, skeleton