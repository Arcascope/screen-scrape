import traceback
import pytesseract
import os
import matplotlib.pyplot as plt
from utils import *

import cv2
import numpy as np


class ImageProcessingError(Exception):
    """Custom exception for image processing failures."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


def load_and_validate_image(filename):
    """Load an image and validate its existence."""
    img = cv2.imread(filename)
    img = convert_dark_mode(img)
    if WANT_DEBUG_SUBIMAGE:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    if img is None:
        raise ImageProcessingError("Failed to load image.")
    return img


def calculate_roi_from_clicks(upper_left, lower_right, snap_to_grid_func=None, img=None):
    """Calculate the region of interest based on user input, optionally snapping to grid."""
    roi_width = lower_right[0] - upper_left[0]
    roi_height = lower_right[1] - upper_left[1]

    if roi_width <= 0 or roi_height <= 0:
        raise ImageProcessingError("Invalid region of interest dimensions.")

    # TODO: Re-add snap-to-grid capabilities
    # if snap_to_grid_func and img is not None:
    #     return snap_to_grid_func(img, upper_left[0], upper_left[1], roi_width, roi_height)
    # else:
    return upper_left[0], upper_left[1], roi_width, roi_height


def process_image_with_grid(filename, upper_left, lower_right, is_battery=False, snap_to_grid=False):
    try:
        img = load_and_validate_image(filename)
        img_copy = img.copy()

        img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)

        if snap_to_grid:
            snap_func = snap_to_grid
        else:
            snap_func = None

        print("Calculating region of interest from clicks...")
        upper_left_x, upper_left_y, roi_width, roi_height = calculate_roi_from_clicks(upper_left, lower_right,
                                                                                      snap_func, img)

        if upper_left_x < 0 or upper_left_y < 0:
            raise ImageProcessingError(
                "ROI coordinates are out of image bounds.")

        roi_x = upper_left_x
        roi_y = upper_left_y

        if is_battery:
            date_in_image = find_time(
                img_copy, roi_x, roi_y, roi_width, roi_height)
        else:
            date_in_image = find_date_in_image(img)
        print("Extracting title...")
        title = find_screenshot_title(img)

        filename, row, graph_filename = save_image(
            filename, roi_x, roi_y, roi_width, roi_height, is_battery)
        return filename, graph_filename, row, title, date_in_image

    except ImageProcessingError as e:
        print(f"Error: {traceback.format_exc()}")
        return None, None, list(range(25)), "", ""


def process_image(filename, is_battery=False, snap_to_grid=False):
    try:
        img = load_and_validate_image(filename)
        return apply_processing(filename, img, is_battery, snap_to_grid)
    except Exception as e:
        print(
            f"Error during image loading or processing: {traceback.format_exc()}")
        return None, None, list(range(25)), "", ""


def adjust_anchor_offsets(d_right, offset):
    for i in range(len(d_right['level'])):
        d_right['left'][i] += offset


def apply_processing(filename, img, is_battery, snap_to_grid):
    try:
        img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)
        img_copy = img.copy()
        img_left, img_right, right_offset = prepare_image_chunks(img)

        # Perform OCR and adjust offsets
        d_left, d_right = perform_ocr(img_left, img_right)
        adjust_anchor_offsets(d_right, right_offset)

        # Detect anchors
        found_12, lower_left_x, lower_left_y = find_left_anchor(
            d_left, img, img_copy)
        found_60, upper_right_x, upper_right_y = find_right_anchor(
            d_right, img, img_copy)

        if not found_12 or not found_60:
            raise ValueError("Couldn't find graph anchors!")

        # Calculate and possibly snap to region of interest
        roi_x, roi_y, roi_width, roi_height = calculate_roi(
            lower_left_x, upper_right_y, upper_right_x -
            lower_left_x, lower_left_y - upper_right_y,
            snap_to_grid, img)

        if is_battery:
            date_in_image = find_time(
                img_copy, roi_x, roi_y, roi_width, roi_height)
        else:
            date_in_image = find_date_in_image(img)

        title = find_screenshot_title(img)

        filename, row, graph_filename = save_image(
            filename, roi_x, roi_y, roi_width, roi_height, is_battery)
        return filename, graph_filename, row, title, date_in_image

    except Exception:
        print(f"Processing failed: {traceback.format_exc()}")
        return None, None, [], "", ""


def find_time(img, roi_x, roi_y, roi_width, roi_height):
    text1, text2, is_pm = get_text(img, roi_x, roi_y, roi_width, roi_height)

    return text1


def prepare_image_chunks(img):
    img_chunk_num = 3
    img_width, img_height = img.shape[1], img.shape[0]
    # Removing top 10% of the image to minimize risk of failed grid detection
    top_removal = int(img_height * 0.10)

    # Split image into left and right for anchor detection
    img_left = img[:, :int(img_width / img_chunk_num)]
    img_right = img[:, -int(img_width / img_chunk_num):]

    # Overwrite the top part of the image to simplify the image for OCR
    img_left[0:top_removal, :] = get_pixel(img_left, 1)
    img_right[0:top_removal, :] = get_pixel(img_right, 1)
    right_offset = img_width - int(img_width / img_chunk_num)

    return img_left, img_right, right_offset


def perform_ocr(img_left, img_right):
    d_left = pytesseract.image_to_data(
        img_left, config='--psm 12', output_type=Output.DICT)
    d_right = pytesseract.image_to_data(
        img_right, config='--psm 12', output_type=Output.DICT)
    return d_left, d_right


def calculate_roi(lower_left_x, upper_right_y, roi_width, roi_height, snap_to_grid, img):
    if snap_to_grid:
        lower_left_x, upper_right_y, roi_width, roi_height = snap_to_grid(
            img, lower_left_x, upper_right_y, roi_width, roi_height)

    if lower_left_x < 0:
        raise ValueError(
            f"Invalid ROI lower left x coordinate: {lower_left_x}")
    elif upper_right_y < 0:
        raise ValueError(
            f"Invalid ROI upper right y coordinate: {upper_right_y}")
    elif roi_width < 0:
        raise ValueError(f"Invalid ROI width value: {roi_width}")
    elif roi_height < 0:
        raise ValueError(f"Invalid ROI height value: {roi_height}")

    return lower_left_x, upper_right_y, roi_width, roi_height


def save_image(filename, roi_x, roi_y, roi_width, roi_height, is_battery):
    print("Preparing to extract grid")
    img = load_and_validate_image(filename)
    img_copy = img.copy()

    if is_battery:
        print("Removing all but the dark blue color...")
        img_new = remove_all_but(img_copy, [255, 121, 0])
        no_dark_blue_detected = np.sum(
            255 - img_new[roi_y:roi_y + roi_height, roi_x: roi_x + roi_width]) < 10
        if no_dark_blue_detected:
            print("No dark blue color detected; assuming dark mode...")
            img_copy = img.copy()
            img_new = remove_all_but(img_copy, [0, 255 - 121, 255])
        img = img_new

    row, img, scale_amount = slice_image(
        img, roi_x, roi_y, roi_width, roi_height)

    print("Saving processed image...")
    save_path = save_processed_image(
        img, roi_x, roi_y, roi_width, roi_height, filename, scale_amount)

    plt.close()
    plt.figure(figsize=(8, 2.25))
    x = range(len(row[:-1]))
    height = row[:-1]
    plt.bar(np.array(x) + 0.5, height)
    plt.ylim([0, 60])
    plt.xlim([0, 24])

    plt.grid(True)
    plt.xticks(ticks=range(0, len(row[:-1]), 6))  # Set x-ticks every 6 units
    plt.yticks(ticks=range(0, 61, 15))  # Set y-ticks every 15 units
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.grid(which='both', linestyle='--', linewidth=0.5)

    debug_folder = 'debug'
    os.makedirs(debug_folder, exist_ok=True)
    graph_save_name = os.path.join(
        debug_folder, "graph" + os.path.basename(filename))
    graph_save_name = graph_save_name.replace(".jfif", ".jpg")

    plt.savefig(graph_save_name, bbox_inches='tight', pad_inches=0)

    return save_path, row, graph_save_name


def save_processed_image(image, x, y, width, height, original_filename, scale_amount):
    """ Save the ROI of the image to a file in the debug directory. """
    debug_folder = 'debug'
    os.makedirs(debug_folder, exist_ok=True)
    save_name = os.path.join(debug_folder, os.path.basename(original_filename))
    roi = image[scale_amount * y:scale_amount * y + scale_amount * height,
                scale_amount * x: scale_amount * x + scale_amount * width]

    save_name = save_name.replace(".jfif", ".jpg")
    cv2.imwrite(save_name, roi)
    return save_name
