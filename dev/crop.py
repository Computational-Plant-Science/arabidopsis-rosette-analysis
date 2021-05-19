import cv2
import numpy as np


# Detect circles in the image
def circle_detect(image_path, template_path):
    template = cv2.imread(template_path, 0)

    # load the image, clone it for output, and then convert it to grayscale
    img_ori = cv2.imread(image_path)

    img_rgb = img_ori.copy()

    # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Store width and height of template in w and h
    w, h = template.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold
    threshold = 0.8

    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)

    if len(loc):

        (y, x) = np.unravel_index(res.argmax(), res.shape)

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)

        # print(y,x)

        # print(min_val, max_val, min_loc, max_loc)

        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            circle_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

            # save segment result
        # result_file = (save_path + base_name + '_circle.' + args['filetype'])
        # print(result_file)
        # cv2.imwrite(result_file, circle_overlay)

        crop_img = img_rgb[y + 150:y + 850, x - 650:x]

        # save segment result
        # result_file = (save_path + base_name + '_cropped.' + args['filetype'])
        # print(result_file)
        # cv2.imwrite(result_file, crop_img)

    return crop_img