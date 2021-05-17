from PIL import Image, ImageEnhance
import cv2


def image_enhance(image_file):
    im = Image.fromarray(cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB))

    im_sharpness = ImageEnhance.Sharpness(im).enhance(3.5)

    im_contrast = ImageEnhance.Contrast(im_sharpness).enhance(1.5)

    im_out = ImageEnhance.Brightness(im_contrast).enhance(1.2)

    return im_out


def increase_brightness(img, value=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


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