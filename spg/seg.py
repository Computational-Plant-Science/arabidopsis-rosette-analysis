import warnings
from collections import Counter
from os.path import join

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.spatial import distance as dist
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed
from sklearn.cluster import KMeans

from spg.threshold import otsu_threshold
from spg.utils import get_cmap, rgb2hex


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


def watershed_seg(thresh, min_distance_value):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this distance map
    D = ndimage.distance_transform_edt(thresh)

    localMax = peak_local_max(D, indices=False, min_distance=min_distance_value, labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    labels = watershed(-D, markers, mask=thresh)

    print(f"{len(np.unique(labels)) - 1} unique segments found")
    return labels


def individual_object_seg(orig, labels, save_path, base_name, file_extension, leaf_images: bool = True):
    (width, height, n_channel) = orig.shape

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros((width, height), dtype="uint8")
        mask[labels == label] = 255

        # apply individual object mask
        masked = cv2.bitwise_and(orig, orig, mask=mask)

        if leaf_images:
            result_img_path = (save_path + base_name + '_leaf_' + str(label) + file_extension)
            cv2.imwrite(result_img_path, masked)


def comp_external_contour(orig, thresh):
    # find contours and get the external one
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width, img_channels = orig.shape

    index = 1

    for c in contours:

        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        if w > img_width * 0.01 and h > img_height * 0.01:

            trait_img = cv2.drawContours(orig, contours, -1, (255, 255, 0), 1)

            # draw a green rectangle to visualize the bounding rect
            roi = orig[y:y + h, x:x + w]

            print("ROI {} detected ...\n".format(index))
            # result_file = (save_path +  str(index) + file_extension)
            # cv2.imwrite(result_file, roi)

            trait_img = cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 255, 0), 3)

            index += 1

            '''
            #get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            #draw a red 'nghien' rectangle
            trait_img = cv2.drawContours(orig, [box], 0, (0, 0, 255))
            '''
            # get convex hull
            hull = cv2.convexHull(c)
            # draw it in red color
            trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255), 3)

            '''
            # calculate epsilon base on contour's perimeter
            # contour's perimeter is returned by cv2.arcLength
            epsilon = 0.01 * cv2.arcLength(c, True)
            # get approx polygons
            approx = cv2.approxPolyDP(c, epsilon, True)
            # draw approx polygons
            trait_img = cv2.drawContours(orig, [approx], -1, (0, 255, 0), 1)
         
            # hull is convex shape as a polygon
            hull = cv2.convexHull(c)
            trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255))
            '''

            '''
            #get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            # and draw the circle in blue
            trait_img = cv2.circle(orig, center, radius, (255, 0, 0), 2)
            '''

            area = cv2.contourArea(c)
            print("Leaf area = {0:.2f}... \n".format(area))

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            print("solidity = {0:.2f}... \n".format(solidity))

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            trait_img = cv2.circle(orig, extLeft, 3, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, extRight, 3, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, extTop, 3, (255, 0, 0), -1)
            trait_img = cv2.circle(orig, extBot, 3, (255, 0, 0), -1)

            max_width = dist.euclidean(extLeft, extRight)
            max_height = dist.euclidean(extTop, extBot)

            if max_width > max_height:
                trait_img = cv2.line(orig, extLeft, extRight, (0, 255, 0), 2)
            else:
                trait_img = cv2.line(orig, extTop, extBot, (0, 255, 0), 2)

            print("Width and height are {0:.2f},{1:.2f}... \n".format(w, h))

    return trait_img, area, solidity, w, h


def color_region(image, mask, output_directory, file_name, num_clusters):
    # read the image
    # grab image width and height
    (h, w) = image.shape[:2]

    # apply the mask to get the segmentation of plant
    masked_image_ori = cv2.bitwise_and(image, image, mask=mask)

    # define result path for labeled images
    result_img_path = join(output_directory, f"{file_name}.masked.png")
    cv2.imwrite(result_img_path, masked_image_ori)

    # convert to RGB
    image_RGB = cv2.cvtColor(masked_image_ori, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_RGB.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    # num_clusters = 5
    compactness, labels, (centers) = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels_flat = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels_flat]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image_RGB.shape)

    segmented_image_BRG = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    # define result path for labeled images
    result_img_path = join(output_directory, f"{file_name}.clustered.png")
    cv2.imwrite(result_img_path, segmented_image_BRG)

    '''
    fig = plt.figure()
    ax = Axes3D(fig)        
    for label, pix in zip(labels, segmented_image):
        ax.scatter(pix[0], pix[1], pix[2], color = (centers))
            
    result_file = (save_path + base_name + 'color_cluster_distributation.png')
    plt.savefig(result_file)
    '''
    # Show only one chosen cluster
    # masked_image = np.copy(image)
    masked_image = np.zeros_like(image_RGB)

    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to render
    # cluster = 2

    cmap = get_cmap(num_clusters + 1)

    # clrs = sns.color_palette('husl', n_colors = num_clusters)  # a list of RGB tuples

    color_conversion = interp1d([0, 1], [0, 255])

    for cluster in range(num_clusters):

        print("Processing Cluster{0} ...\n".format(cluster))
        # print(clrs[cluster])
        # print(color_conversion(clrs[cluster]))

        masked_image[labels_flat == cluster] = centers[cluster]

        # print(centers[cluster])

        # convert back to original shape
        masked_image_rp = masked_image.reshape(image_RGB.shape)

        # masked_image_BRG = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('maksed.png', masked_image_BRG)

        gray = cv2.cvtColor(masked_image_rp, cv2.COLOR_BGR2GRAY)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # thresh_cleaned = clear_border(thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)

        '''
        # compute the center of the contour area and draw a circle representing the center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        result = cv2.putText(masked_image_rp, "#{}".format(cluster + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        '''

        if not cnts:
            print("findContours is empty")
        else:

            # loop over the (unsorted) contours and draw them
            for (i, c) in enumerate(cnts):
                result = cv2.drawContours(masked_image_rp, c, -1, color_conversion(np.random.random(3)), 2)
                # result = cv2.drawContours(masked_image_rp, c, -1, color_conversion(clrs[cluster]), 2)

            # result = result(np.where(result == 0)== 255)
            result[result == 0] = 255

            result_BRG = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            result_img_path = join(output_directory, f"{file_name}.result.{cluster}.png")
            cv2.imwrite(result_img_path, result_BRG)

    counts = Counter(labels_flat)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = centers

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # print(hex_colors)

    index_bkg = [index for index in range(len(hex_colors)) if hex_colors[index] == '#000000']

    # print(index_bkg[0])

    # print(counts)
    # remove background color
    del hex_colors[index_bkg[0]]
    del rgb_colors[index_bkg[0]]

    # Using dictionary comprehension to find list
    # keys having value .
    delete = [key for key in counts if key == index_bkg[0]]

    # delete the key
    for key in delete: del counts[key]

    fig = plt.figure(figsize=(6, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

    # define result path for labeled images
    result_img_path = join(output_directory, f"{file_name}.pie_color.png")
    plt.savefig(result_img_path)

    return rgb_colors