import os
import traceback
import warnings
from contextlib import closing
from multiprocessing import Pool
from os.path import join
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skan import Skeleton, summarize, draw
from skimage import img_as_ubyte
from skimage.color import rgb2lab, deltaE_cie76

from spg.curvature import compute_curv
from spg.popos import SPGOptions, TraitsResult, InputImage
from spg.seg import color_cluster_seg, watershed_seg, individual_object_seg, comp_external_contour, color_region
from spg.skel import skeleton_bw
from spg.utils import outlier_double_mad

warnings.filterwarnings("ignore")

MB_FACTOR = float(1 << 20)


def extract_traits_from_image(input_image: InputImage, output_directory: str):
    try:
        _, file_extension = os.path.splitext(input_image.name)
        file_size = os.path.getsize(input_image.path) / MB_FACTOR

        print("Segmenting plant object using automatic color clustering method")
        if (file_size > 5.0):
            print(f"It may take some time due to large file size ({file_size} MB)")

        args_colorspace = 'lab'
        args_channels = '1'
        args_num_clusters = 2

        image = cv2.imread(input_image.path)

        # circle detection
        # _, circles, cropped = circle_detect(options)
        image_copy = image.copy()

        # color clustering based plant object segmentation
        segmented = color_cluster_seg(image_copy, args_colorspace, args_channels, args_num_clusters)
        cv2.imwrite(join(output_directory, f"{input_image.stem}_seg{file_extension}"), segmented)

        num_clusters = 5
        # save color quantization result
        # rgb_colors = color_quantization(image, thresh, save_path, num_clusters)
        rgb_colors = color_region(image_copy, segmented, output_directory + '/', input_image.stem, num_clusters)

        selected_color = rgb2lab(np.uint8(np.asarray([[rgb_colors[0]]])))

        print("Color difference are : ")

        print(selected_color)

        color_diff = []

        for index, value in enumerate(rgb_colors):
            # print(index, value)
            curr_color = rgb2lab(np.uint8(np.asarray([[value]])))
            diff = deltaE_cie76(selected_color, curr_color)

            color_diff.append(diff)

            print(index, value, diff)

            ###############################################

        # accquire medial axis of segmentation mask
        # image_medial_axis = medial_axis_image(thresh)

        image_skeleton, skeleton = skeleton_bw(segmented)

        # save _skeleton result
        cv2.imwrite(join(output_directory, f"{input_image.stem}_skeleton{file_extension}"), img_as_ubyte(image_skeleton))

        ###
        # ['skeleton-id', 'node-id-src', 'node-id-dst', 'branch-distance',
        # 'branch-type', 'mean-pixel-value', 'stdev-pixel-value',
        # 'image-coord-src-0', 'image-coord-src-1', 'image-coord-dst-0', 'image-coord-dst-1',
        # 'coord-src-0', 'coord-src-1', 'coord-dst-0', 'coord-dst-1', 'euclidean-distance']
        ###

        # get brach data
        branch_data = summarize(Skeleton(image_skeleton))
        # print(branch_data)

        # select end branch
        sub_branch = branch_data.loc[branch_data['branch-type'] == 1]

        sub_branch_branch_distance = sub_branch["branch-distance"].tolist()

        # remove outliers in branch distance
        outlier_list = outlier_double_mad(sub_branch_branch_distance, thresh=3.5)

        indices = [i for i, x in enumerate(outlier_list) if x]

        sub_branch_cleaned = sub_branch.drop(sub_branch.index[indices])

        # print(outlier_list)
        # print(indices)
        # print(sub_branch)

        print(sub_branch_cleaned)

        '''
        min_distance_value_list = sub_branch_cleaned["branch-distance"].tolist()

        min_distance_value_list.sort()

        min_distance_value = int(min_distance_value_list[2])

        print("Smallest branch-distance is:", min_distance_value)

        #fig = plt.plot()

        (img_endpt_overlay, img_marker) = overlay_skeleton_endpoints(source_image, sub_branch_cleaned)

        result_file = (save_path + base_name + '_endpts_overlay' + file_extension)
        plt.savefig(result_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        plt.close()

        result_file = (save_path + base_name + '_marker' + file_extension)
        cv2.imwrite(result_file, img_marker)
        '''

        branch_type_list = sub_branch_cleaned["branch-type"].tolist()

        # print(branch_type_list.count(1))

        print("[INFO] {} branch end points found\n".format(branch_type_list.count(1)))

        # img_hist = branch_data.hist(column = 'branch-distance', by = 'branch-type', bins = 100)
        # result_file = (save_path + base_name + '_hist' + file_extension)
        # plt.savefig(result_file, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        # plt.close()

        fig = plt.plot()
        source_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        # img_overlay = draw.overlay_euclidean_skeleton_2d(source_image, branch_data, skeleton_color_source = 'branch-distance', skeleton_colormap = 'hsv')
        img_overlay = draw.overlay_euclidean_skeleton_2d(source_image, branch_data, skeleton_color_source='branch-type', skeleton_colormap='hsv')
        plt.savefig(join(output_directory, f"{input_image.stem}_euclidean_graph_overlay{file_extension}"), transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        ############################################## leaf number computation
        min_distance_value = 20
        # watershed based leaf area segmentaiton
        labels = watershed_seg(segmented, min_distance_value)

        # labels = watershed_seg_marker(orig, thresh, min_distance_value, img_marker)

        individual_object_seg(image_copy, labels, output_directory + '/', input_image.stem, file_extension)

        # save watershed result label image
        # Map component labels to hue val
        label_hue = np.uint8(128 * labels / np.max(labels))
        # label_hue[labels == largest_label] = np.uint8(15)
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set background label to black
        labeled_img[label_hue == 0] = 0
        # plt.imsave(result_file, img_as_float(labels), cmap = "Spectral")
        cv2.imwrite(join(output_directory, f"{input_image.stem}_label{file_extension}"), labeled_img)

        (avg_curv, label_trait) = compute_curv(image_copy, labels)

        # save watershed result label image
        cv2.imwrite(join(output_directory, f"{input_image.stem}_curv{file_extension}"), label_trait)

        # find external contour
        (trait_img, area, solidity, max_width, max_height) = comp_external_contour(image_copy, segmented)
        # save segmentation result
        # print(filename)
        cv2.imwrite(join(output_directory, f"{input_image.stem}_excontour{file_extension}"), trait_img)

        n_leaves = int(len(np.unique(labels)) / 1 - 1)

        # print("[INFO] {} n_leaves found\n".format(len(np.unique(labels)) - 1))

        # Path("/tmp/d/a.dat").name

        return TraitsResult(name=input_image.stem, failed=False, area=area, solidity=solidity, max_width=max_width, max_height=max_height,
                            avg_curvature=avg_curv, leaves=n_leaves)
    except:
        print(f"Error in trait extraction: {traceback.format_exc()}")
        return TraitsResult(input_image.stem, True, None, None, None, None, None, None)


def extract_traits(options: SPGOptions) -> List[TraitsResult]:
    if options.multiprocessing:
        cpus = os.cpu_count()
        print(f"Using up to {cpus} processes to extract traits from {len(options.input_images)} image(s)")
        with closing(Pool(processes=cpus)) as pool:
            traits_results = pool.starmap(extract_traits_from_image, [(img, options.output_directory) for img in options.input_images])
            pool.terminate()
    else:
        print(f"Using a single process to extract traits from {len(options.input_images)} image(s)")
        traits_results = [extract_traits_from_image(img, options.output_directory) for img in options.input_images]

    return traits_results
