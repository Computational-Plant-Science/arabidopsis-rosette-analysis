# def overlay_skeleton_endpoints(image, stats, *, image_cmap=None, axes=None):
#     image = _normalise_image(image, image_cmap=image_cmap)
#     summary = stats
#     # transforming from row, col to x, y
#     # coords_cols = (['image-coord-src-%i' % i for i in [1, 0]] +
#     #               ['image-coord-dst-%i' % i for i in [1, 0]])
#
#     coords_cols_src = (['image-coord-src-%i' % i for i in [1, 0]])
#     coords_cols_dst = (['image-coord-dst-%i' % i for i in [1, 0]])
#
#     # coords = summary[coords_cols].values.reshape((-1, 1, 2))
#
#     coords_src = summary[coords_cols_src].values
#     coords_dst = summary[coords_cols_dst].values
#
#     coords_src_x = [i[0] for i in coords_src]
#     coords_src_y = [i[1] for i in coords_src]
#
#     coords_dst_x = [i[0] for i in coords_dst]
#     coords_dst_y = [i[1] for i in coords_dst]
#
#     img_marker = np.zeros_like(image, dtype=np.uint8)
#     img_marker.fill(0)  # or img[:] = 255
#     img_marker[list(map(int, coords_src_y)), list(map(int, coords_src_x))] = 255
#     img_marker[list(map(int, coords_dst_y)), list(map(int, coords_dst_x))] = 255
#
#     # print("img_marker")
#     # print(img_marker.shape)
#
#     if axes is None:
#         fig, axes = plt.subplots()
#
#     axes.axis('off')
#     axes.imshow(image)
#
#     axes.scatter(coords_src_x, coords_src_y, c='w')
#     axes.scatter(coords_dst_x, coords_dst_y, c='w')
#
#     return fig, img_marker
#     # return coords


# def _normalise_image(image, *, image_cmap=None):
#     image = img_as_float(image)
#     if image.ndim == 2:
#         if image_cmap is None:
#             image = gray2rgb(image)
#         else:
#             image = plt.get_cmap(image_cmap)(image)[..., :3]
#     return image


# def color_quantization(image, mask, save_path, num_clusters):
#     # grab image width and height
#     (h, w) = image.shape[:2]
#
#     # change the color storage order
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # apply the mask to get the segmentation of plant
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#
#     # reshape the image to be a list of pixels
#     pixels = masked_image.reshape((masked_image.shape[0] * masked_image.shape[1], 3))
#
#     ############################################################
#     # Clustering process
#     ###############################################################
#     # cluster the pixel intensities
#     clt = MiniBatchKMeans(n_clusters=num_clusters)
#     # clt = KMeans(n_clusters = args["clusters"])
#     clt.fit(pixels)
#
#     # assign labels to each cluster
#     labels = clt.fit_predict(pixels)
#
#     # obtain the quantized clusters using each label
#     quant = clt.cluster_centers_.astype("uint8")[labels]
#
#     # reshape the feature vectors to images
#     quant = quant.reshape((h, w, 3))
#     image_rec = pixels.reshape((h, w, 3))
#
#     # convert from L*a*b* to RGB
#     quant = cv2.cvtColor(quant, cv2.COLOR_RGB2BGR)
#     image_rec = cv2.cvtColor(image_rec, cv2.COLOR_RGB2BGR)
#
#     # display the images
#     # cv2.imshow("image", np.hstack([image_rec, quant]))
#     # cv2.waitKey(0)
#
#     # define result path for labeled images
#     result_img_path = save_path + 'cluster_out.png'
#
#     # save color_quantization results
#     cv2.imwrite(result_img_path, quant)
#
#     # Get colors and analze them from masked image
#     counts = Counter(labels)
#     # sort to ensure correct color percentage
#     counts = dict(sorted(counts.items()))
#
#     center_colors = clt.cluster_centers_
#
#     # print(type(center_colors))
#
#     # We get ordered colors by iterating through the keys
#     ordered_colors = [center_colors[i] for i in counts.keys()]
#     hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
#     rgb_colors = [ordered_colors[i] for i in counts.keys()]
#
#     #######################################################################################
#     threshold = 60
#
#     selected_color = rgb2lab(np.uint8(np.asarray([[rgb_colors[0]]])))
#
#     for i in range(num_clusters):
#         curr_color = rgb2lab(np.uint8(np.asarray([[rgb_colors[i]]])))
#         diff = deltaE_cie76(selected_color, curr_color)
#         if (diff < threshold):
#             print("Color difference value is : {0} \n".format(str(diff)))
#     ###########################################################################################
#     # print(hex_colors)
#
#     index_bkg = [index for index in range(len(hex_colors)) if hex_colors[index] == '#000000']
#
#     # print(index_bkg[0])
#
#     # print(counts)
#     # remove background color
#     del hex_colors[index_bkg[0]]
#     del rgb_colors[index_bkg[0]]
#
#     # Using dictionary comprehension to find list
#     # keys having value .
#     delete = [key for key in counts if key == index_bkg[0]]
#
#     # delete the key
#     for key in delete: del counts[key]
#
#     fig = plt.figure(figsize=(6, 6))
#     plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
#
#     # define result path for labeled images
#     result_img_path = save_path + 'pie_color.png'
#     plt.savefig(result_img_path)
#
#     # build a histogram of clusters and then create a figure representing the number of pixels labeled to each color
#     hist = utils.centroid_histogram(clt)
#
#     # remove the background color cluster
#     clt.cluster_centers_ = np.delete(clt.cluster_centers_, index_bkg[0], axis=0)
#
#     # build a histogram of clusters using center lables
#     numLabels = utils.plot_centroid_histogram(save_path, clt)
#
#     # create a figure representing the distribution of each color
#     bar = utils.plot_colors(hist, clt.cluster_centers_)
#
#     # save a figure of color bar
#     utils.plot_color_bar(save_path, bar)
#
#     return rgb_colors


# def watershed_seg_marker(orig, thresh, min_distance_value, img_marker):
#     # compute the exact Euclidean distance from every binary
#     # pixel to the nearest zero pixel, then find peaks in this
#     # distance map
#     D = ndimage.distance_transform_edt(thresh)
#
#     gray = cv2.cvtColor(img_marker, cv2.COLOR_BGR2GRAY)
#     img_marker = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
#     # localMax = peak_local_max(D, indices = False, min_distance = min_distance_value,  labels = thresh)
#
#     # perform a connected component analysis on the local peaks,
#     # using 8-connectivity, then appy the Watershed algorithm
#     markers = ndimage.label(img_marker, structure=np.ones((3, 3)))[0]
#
#     labels = watershed(-D, markers, mask=thresh)
#
#     props = regionprops(labels)
#
#     areas = [p.area for p in props]
#
#     import statistics
#
#     # outlier_list = outlier_doubleMAD(areas, thresh = 1.0)
#
#     # indices = [i for i, x in enumerate(outlier_list) if x]
#
#     print(areas)
#     print(statistics.mean(areas))
#     #
#     # print(outlier_list)
#     # print(indices)
#
#     print("[INFO] {} unique segments found\n".format(len(np.unique(labels)) - 1))
#
#     return labels


# def medial_axis_image(thresh):
#     # convert an image from OpenCV to skimage
#     thresh_sk = img_as_float(thresh)
#
#     image_bw = img_as_bool((thresh_sk))
#
#     image_medial_axis = medial_axis(image_bw)
#
#     return image_medial_axis