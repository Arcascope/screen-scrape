import cv2
import pytesseract
import numpy as np


WANT_DEBUG_LINE_FIND = False
WANT_DEBUG_GRID = True
WANT_DEBUG_SUBIMAGE = True
WANT_DEBUG_TITLE = False
WANT_DEBUG_SLICE = False
WANT_DEBUG_TEXT = False

VERBOSE = True


def show_until_destroyed(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)

def extract_date(image):
    text = pytesseract.image_to_string(image)

    return text


def get_pixel(img, arg):
    '''Get the dominant pixel in the image'''
    unq, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    sort = np.argsort(count)
    sorted_unq = unq[sort]
    if np.abs(arg) > len(sorted_unq):
        return sorted_unq[0]
    return sorted_unq[arg]


def is_close(pixel_1, pixel_2, thresh=1):
    '''Decide if two pixels are close enough'''
    if np.sum(np.abs(pixel_1 - pixel_2)) <= thresh * len(pixel_1):
        return True
    return False


def darken_non_white(img):
    '''Darken the non-white pixels in the bars'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img[thresh < 250] = 0
    return img


def reduce_color_count(img, num_colors):
    '''Reduce the color count to help with aliasing'''
    for i in range(num_colors):
        img[(img >= i * 255 / num_colors) & (img < (i + 1) * 255 / num_colors)] = i * 255 / (num_colors - 1)
    return img


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def extract_line(img, x0, x1, y0, y1, mode, threshold=20):
    sub_image = img[y0:y1, x0:x1].copy()

    # Binarize for line extraction
    shape = np.shape(sub_image)

    dark_mode_threshold = 100
    if np.mean(img) < dark_mode_threshold:
        pixel_to_target = [49, 45, 44]
        background_color = [0, 0, 0]
        opposite_color = [255, 255, 255]
    else:
        pixel_to_target = [230, 230, 230]
        background_color = [255, 255, 255]
        opposite_color = [0, 0, 0]

    if WANT_DEBUG_SUBIMAGE:
        cv2.imshow('img', sub_image)
        cv2.waitKey(0)

    # Binarize
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = sub_image[i, j]
            print(pixel)
            if is_close(pixel, pixel_to_target, threshold):
                sub_image[i, j] = opposite_color
            else:
                sub_image[i, j] = background_color

    count_color = opposite_color

    if WANT_DEBUG_SUBIMAGE:
        cv2.imshow('img', sub_image)
        cv2.waitKey(0)
    if mode == "horizontal":
        shape = np.shape(sub_image)

        for i in range(shape[0]):
            row_score = 0
            for j in range(shape[1]):
                pixel = sub_image[i, j]
                if is_close(pixel, count_color):
                    row_score = row_score + 1
            if row_score > 0.5 * shape[1]:  # Threshold set by inspection; can be modified
                return i

    if mode == "vertical":
        shape = np.shape(sub_image)
        for j in range(shape[1]):
            col_score = 0
            for i in range(shape[0]):
                pixel = sub_image[i, j]
                if is_close(pixel, count_color):
                    col_score = col_score + 1

            if col_score > 0.3 * shape[0]:
                return j
