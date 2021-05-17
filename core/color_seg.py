import warnings

import cv2
import numpy as np
from skimage.segmentation import clear_border
from sklearn.cluster import KMeans

from core.thresholding import otsu_threshold

warnings.filterwarnings("ignore")

MBFACTOR = float(1<<20)


def color_cluster_seg(image, args_colorspace, args_channels, args_num_clusters):
    """
    K-means color clustering based segmentation. This is achieved
    by converting the source image to a desired color space and
    running K-means clustering on only the desired channels,
    with the pixels being grouped into a desired number of clusters.
    """

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep only the selected channels for K-means clustering.
    if args_channels != 'all':
        channels = cv2.split(image)
        channelIndices = []
        for char in args_channels:
            channelIndices.append(int(char))
        image = image[:, :, channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)

    (width, height, n_channel) = image.shape

    # print("image shape: \n")
    # print(width, height, n_channel)

    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')

    # define number of cluster
    numClusters = max(2, args_num_clusters)

    # clustering method
    kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

    # get lables
    pred_label = kmeans.labels_

    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)], key=lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

    thresh = otsu_threshold(kmeansImage)

    if np.count_nonzero(thresh) > 0:

        thresh_cleaned_bw = clear_border(thresh)
    else:
        thresh_cleaned_bw = thresh

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh_cleaned_bw, connectivity=8)

    # stats[0], centroids[0] are for the background label. ignore
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    sizes = stats[1:, cv2.CC_STAT_AREA]

    Coord_left = stats[1:, cv2.CC_STAT_LEFT]

    Coord_top = stats[1:, cv2.CC_STAT_TOP]

    Coord_width = stats[1:, cv2.CC_STAT_WIDTH]

    Coord_height = stats[1:, cv2.CC_STAT_HEIGHT]

    Coord_centroids = centroids

    # print("Coord_centroids {}\n".format(centroids[1][1]))

    # print("[width, height] {} {}\n".format(width, height))

    nb_components = nb_components - 1

    min_size = 1000

    max_size = width * height * 0.1

    img_thresh = np.zeros([width, height], dtype=np.uint8)

    for i in range(0, nb_components):

        if (sizes[i] >= min_size):

            if (Coord_left[i] > 1) and (Coord_top[i] > 1) and (Coord_width[i] - Coord_left[i] > 0) and (Coord_height[i] - Coord_top[i] > 0) and (
                    centroids[i][0] - width * 0.5 < 10) and (centroids[i][1] - height * 0.5 < 10):
                img_thresh[output == i + 1] = 255
                print("Foreground center found ")

            elif ((Coord_width[i] - Coord_left[i]) * 0.5 - width < 15) and (centroids[i][0] - width * 0.5 < 15) and (
                    centroids[i][1] - height * 0.5 < 15) and ((sizes[i] <= max_size)):
                imax = max(enumerate(sizes), key=(lambda x: x[1]))[0] + 1
                img_thresh[output == imax] = 255
                print("Foreground max found ")

            else:
                img_thresh[output == i + 1] = 255

    return img_thresh