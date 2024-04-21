import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import subprocess

from main import WANT_DEBUG_SUBIMAGE

DEBUG = False

error_state = -1, -1, -1, -1

def show_until_destroyed(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_all_text(image):
    # Increase contrast
    image = adjust_contrast_brightness(image, contrast=2.0, brightness=0)

    # Extract boxes
    dictionary = pytesseract.image_to_data(image, output_type=Output.DICT)

    if DEBUG:
        cv2.imwrite("temp_output.png", image)
        process = subprocess.run(["tesseract", "temp_output.png", "temp.txt", "config.txt"], shell=False,
                                 stdout=subprocess.PIPE)

    # Run multiple passes at catching text with OCR; can expand range
    for i in range(13, 14):
        dictionary_temp = pytesseract.image_to_data(image, config='--psm ' + str(i), output_type=Output.DICT)
        dictionary = dictionary_temp | dictionary

    return dictionary


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


def remove_line_color(img):
    line_color = np.array([203, 199, 199])
    shape = np.shape(img)

    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = img[i, j]
            if is_close(pixel, line_color):
                img[i, j] = [255, 255, 255]

    return img


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


def extract_date(image):
    text = pytesseract.image_to_string(image)
    return text



def extract_line(img, x0, x1, y0, y1, mode, threshold=20):
    sub_image = img[y0:y1, x0:x1].copy()

    # Binarize for line extraction
    shape = np.shape(sub_image)
    sub_image = reduce_color_count(sub_image, 2)
    pixel_to_target = [0, 0, 0]
    background_color = [255, 255, 255]
    opposite_color = [0, 0, 0]

    # dark_mode_threshold = 100
    # if np.mean(img) < dark_mode_threshold:
    #     pixel_to_target = [49, 45, 44]
    #     background_color = [0, 0, 0]
    #     opposite_color = [255, 255, 255]
    # else:
    #     pixel_to_target = [230, 230, 230]
    #     pixel_to_target = [162, 162, 162]
    #     background_color = [255, 255, 255]
    #     opposite_color = [0, 0, 0]

    if WANT_DEBUG_SUBIMAGE:
        cv2.imshow('img', sub_image)
        cv2.waitKey(0)

    # Binarize
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = sub_image[i, j]
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


def snap_to_grid(img, x, y, w, h):
    buffer = 40
    maximum_offset = 20

    line_row = None
    line_col = None

    # Inch down until you find the grid
    moving_index = 0
    while line_row is None and moving_index < maximum_offset:
        line_row = extract_line(img,
                                x,
                                x + buffer,
                                y - buffer + moving_index,
                                y + moving_index, "horizontal")
        moving_index = moving_index + 1
    if line_row is None:
        return error_state

    upper_left_y = y + line_row + moving_index - buffer

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None and moving_index < maximum_offset:
        line_col = extract_line(img,
                                x - buffer + moving_index,
                                x + moving_index,
                                y,
                                y + buffer, "vertical")
        moving_index = moving_index + 1
    if line_col is None:
        return error_state

    upper_left_x = x - buffer + line_col + moving_index

    line_row = None
    line_col = None

    #  Inch down until you find the grid...
    moving_index = 0
    while line_row is None and moving_index < maximum_offset:
        # Look at the gap between the second to last and last hour of the day
        line_row = extract_line(img,
                                x + int(23 * w / 24 - buffer / 2),
                                x + int(23 * w / 24 + buffer / 2),
                                y + h + moving_index - buffer,
                                y + h + moving_index,
                                "horizontal")

        moving_index = moving_index + 1
    if line_row is None:
        return error_state

    lower_right_y = y + h - buffer + line_row + moving_index

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None and moving_index < maximum_offset:
        line_col = extract_line(img,
                                x + w + moving_index - buffer,
                                x + w + moving_index,
                                y + h - buffer,
                                y + h,
                                "vertical")
        moving_index = moving_index + 1
    if line_col is None:
        return error_state

    lower_right_x = x + w - buffer + line_col + moving_index

    return upper_left_x, upper_left_y, lower_right_x - upper_left_x, lower_right_y - upper_left_y
