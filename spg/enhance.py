from PIL import Image, ImageEnhance
import cv2


def image_enhance(image_file):
    im = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))

    im_sharpness = ImageEnhance.Sharpness(im).enhance(3.5)

    im_contrast = ImageEnhance.Contrast(im_sharpness).enhance(1.5)

    im_out = ImageEnhance.Brightness(im_contrast).enhance(1.2)

    return im_out


# apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to perfrom image enhancement
def image_enhance2(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # split on 3 different channels
    l, a, b = cv2.split(lab)

    # apply CLAHE to the L-channel
    l2 = clahe.apply(l)

    # merge channels
    lab = cv2.merge((l2, a, b))

    # convert from LAB to BGR
    img_enhance = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img_enhance


def increase_brightness(img, value=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def gamma_correction(image_file):
    # parse the file name
    path, filename = os.path.split(image_file)

    # filename, file_extension = os.path.splitext(image_file)

    # construct the result file path
    result_img_path = save_path + str(filename[0:-4]) + '.' + ext

    print("Enhancing image : {0} \n".format(str(filename)))

    # Load the image
    image = cv2.imread(image_file)

    # get size of image
    img_height, img_width = image.shape[:2]

    # image = cv2.resize(image, (0,0), fx = scale_factor, fy = scale_factor)

    gamma = args['gamma']

    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1

    adjusted = adjust_gamma(image, gamma=gamma)

    enhanced_image = image_enhance(adjusted)

    # save result as images for reference
    cv2.imwrite(result_img_path, enhanced_image)

# alternate CV2 enhancement:

# clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(3, 3))
# cl = clahe.apply(L)
# lightened = cv2.merge((cl, A, B))
# converted = cv2.cvtColor(lightened, cv2.COLOR_LAB2BGR)

# hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsvImg[..., 2] = hsvImg[..., 2] * 2
# converted = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
# v += 255
# final_hsv = cv2.merge((h, s, v))
# converted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# converted = increase_brightness(converted)

# cv2.imwrite(join(options.output_directory, new_image_file), converted)