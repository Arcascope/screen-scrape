import datetime

import cv2
from pytesseract import pytesseract, Output
import numpy as np
import re

WANT_DEBUG_LINE_FIND = False
WANT_DEBUG_LEFT = False
WANT_DEBUG_SLICE = False
WANT_DEBUG_SUBIMAGE = False
WANT_DEBUG_SUBIMAGE_SNAP = False
WANT_DEBUG_TEXT = False
VERBOSE = True
error_state = -1, -1, -1, -1


def find_closest_date_substring(text):
    # Regex pattern for "Day, Month Day" format
    date_pattern = r"^[A-Za-z]+, [A-Za-z]+ \d{1,2}$"

    def is_valid_date_format(s):
        """Check if a substring matches the general date format."""
        return bool(re.fullmatch(date_pattern, s))

    # Split the input string by spaces and recombine into substrings
    words = text.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substring = " ".join(words[i:j])
            if is_valid_date_format(substring):
                remove_first = substring.split(", ")[-1]
                return remove_first

    return ""  # Return nothing if no valid date format is found

def find_date_in_image(img):
    date_find = pytesseract.image_to_data(img, config='--psm 6', output_type=Output.DICT)
    all_text_combined = ' '.join(date_find['text'])
    return find_closest_date_substring(all_text_combined)

def find_screenshot_title(img):
    title = ""

    title_find = pytesseract.image_to_data(img, config='--psm 6', output_type=Output.DICT)
    info_rect = [40, 300, 120, 750]  # Default title location

    found_info = False
    for i in range(len(title_find["level"])):
        if "INFO" in title_find["text"][i]:
            info_rect = [title_find['left'][i], title_find['top'][i], title_find['width'][i], title_find['height'][i]]
            found_info = True

    if found_info:  # If we successfully found the "info" string...
        # Look for text underneath the info rectangle:
        app_height = info_rect[3] * 7
        title_origin_y = info_rect[1] + info_rect[3]
        x_origin = info_rect[0] + int(1.5 * info_rect[2])
        x_width = x_origin + int(info_rect[2]) * 8
        app_extract = img[title_origin_y:title_origin_y + app_height, x_origin:x_width]
    else:
        # Default to initial value
        app_extract = img[info_rect[0]:info_rect[2], info_rect[1]:info_rect[3]]

    if len(app_extract) > 0:
        # Crop just to area of app name:
        app_find = extract_all_text(app_extract)

        for i in range(len(app_find["level"])):
            (x, y, w, h) = (app_find['left'][i], app_find['top'][i], app_find['width'][i], app_find['height'][i])
            cv2.rectangle(app_extract, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(app_find["text"][i]) > 2:
                title = title + " " + app_find["text"][i]
                print("Found title: " + title)

    title = title.lstrip()

    return title


def extract_all_text(image):
    # Increase contrast
    image = adjust_contrast_brightness(image, contrast=2.0, brightness=0)

    # Extract boxes
    dictionary = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Run multiple passes at catching text with OCR; can expand range
    for i in range(13, 14):
        dictionary_temp = pytesseract.image_to_data(image, config='--psm ' + str(i), output_type=Output.DICT)
        dictionary = dictionary_temp | dictionary

    return dictionary


def convert_dark_mode(img):
    dark_mode_threshold = 100
    if np.mean(img) < dark_mode_threshold:
        img = 255 - img
        img = adjust_contrast_brightness(img, 3.0, 10)

    return img


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def get_pixel(img, arg):
    '''Get the dominant pixel in the image'''
    unq, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    sort = np.argsort(count)
    sorted_unq = unq[sort]
    if len(sorted_unq) <= 1:
        return None
    if np.abs(arg) >= len(sorted_unq):
        return sorted_unq[0]
    return sorted_unq[arg]


def find_right_anchor(d, img, img_copy):
    found_flag = False
    n_boxes = len(d['level'])
    upper_right_x = -1
    upper_right_y = -1
    buffer = 25
    maximum_offset = 100
    key_list = ["60"]

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if any(key in d['text'][i] for key in key_list):
            if not found_flag:
                found_flag = True
                if WANT_DEBUG_LINE_FIND:
                    cv2.rectangle(img_copy,
                                  (x - buffer, y),
                                  (x, y + buffer),
                                  (255, 0, 3),
                                  2)
                    show_until_destroyed("Image", img_copy)

                line_row = None
                line_col = None

                #  Inch up until you find the grid...
                print("Moving up to search for right anchor...")
                moving_index = 0
                while line_row is None and moving_index < maximum_offset:
                    line_row = extract_line(img,
                                            x - buffer,
                                            x,
                                            y - moving_index,
                                            y - moving_index + h + buffer, "horizontal")
                    moving_index = moving_index + 1
                upper_right_y = y + line_row - moving_index

                #  Inch left until you find the grid...
                print("Moving left to search for right anchor...")
                moving_index = 0
                while line_col is None and moving_index < maximum_offset:
                    line_col = extract_line(img,
                                            x - buffer - moving_index,
                                            x - moving_index,
                                            y,
                                            y + h + buffer, "vertical")
                    moving_index = moving_index + 1
                upper_right_x = x - buffer + line_col - moving_index

                if WANT_DEBUG_LINE_FIND:
                    cv2.rectangle(img_copy,
                                  (upper_right_x, upper_right_y),
                                  (upper_right_x + buffer, upper_right_y + buffer),
                                  (0, 0, 3), 2)

                    show_until_destroyed("Image", img_copy)

    return found_flag, upper_right_x, upper_right_y


def find_left_anchor(d, img, img_copy):
    found_flag = False
    n_boxes = len(d['level'])
    lower_left_x = -1
    lower_left_y = -1
    buffer = 25
    key_list = ["2A", "12", "AM"]
    maximum_offset = 100

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if any(key in d['text'][i] for key in key_list):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

            if not found_flag:
                found_flag = True

                if WANT_DEBUG_LINE_FIND:
                    cv2.rectangle(img_copy,
                                  (x - buffer, y - buffer),
                                  (x + buffer + w, y + buffer + h),
                                  (255, 0, 3), 2)
                    show_until_destroyed("Image", img_copy)

                line_row = None
                line_col = None

                #  Inch up until you find the grid...
                print("Moving up to search for left anchor...")
                moving_index = 0
                while line_row is None and moving_index < maximum_offset:
                    line_row = extract_line(img,
                                            x - buffer,
                                            x + w + buffer,
                                            y - moving_index - buffer,
                                            y - moving_index + buffer,
                                            "horizontal")

                    moving_index = moving_index + 1
                lower_left_y = y - buffer + line_row - moving_index

                #  Inch left until you find the grid...
                print("Moving left to search for left anchor...")
                moving_index = 0
                while line_col is None and moving_index < maximum_offset:
                    line_col = extract_line(img,
                                            x - moving_index - buffer,
                                            x - moving_index + buffer,
                                            y - buffer,
                                            y,
                                            "vertical")
                    moving_index = moving_index + 1
                lower_left_x = x - buffer + line_col - moving_index

                if WANT_DEBUG_LINE_FIND:
                    cv2.rectangle(img_copy,
                                  (x - 2 * buffer, y - buffer),
                                  (x - buffer, y + h + buffer),
                                  (0, 255, 255), 2)
                    cv2.rectangle(img_copy,
                                  (lower_left_x, lower_left_y),
                                  (lower_left_x + buffer, lower_left_y + buffer),
                                  (255, 0, 3), 2)
                    show_until_destroyed("Image", img_copy)

    return found_flag, lower_left_x, lower_left_y


def show_until_destroyed(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_line(img, x0, x1, y0, y1, mode):
    sub_image = img[y0:y1, x0:x1]

    # Binarize for line extraction
    sub_image = reduce_color_count(sub_image, 2)
    pixel_value = get_pixel(sub_image, -2)
    if pixel_value is None:
        return None

    if WANT_DEBUG_SUBIMAGE:
        cv2.imshow('img', sub_image)
        cv2.waitKey(0)
    if mode == "horizontal":
        shape = np.shape(sub_image)

        for i in range(shape[0]):
            row_score = 0
            for j in range(shape[1]):
                pixel = sub_image[i, j]
                if is_close(pixel, pixel_value):
                    row_score = row_score + 1
            if row_score > 0.5 * shape[1]:  # Threshold set by inspection; can be modified
                return i

    if mode == "vertical":
        shape = np.shape(sub_image)
        for j in range(shape[1]):
            col_score = 0
            for i in range(shape[0]):
                pixel = sub_image[i, j]
                if is_close(pixel, pixel_value):
                    col_score = col_score + 1

            if col_score > 0.25 * shape[0]:  # Threshold set by inspection; can be modified
                return j


def is_close(pixel_1, pixel_2, thresh=1):
    '''Decide if two pixels are close enough'''
    if np.sum(np.abs(pixel_1 - pixel_2)) <= thresh * len(pixel_1):
        return True
    return False


def reduce_color_count(img, num_colors):
    '''Reduce the color count to help with aliasing'''
    for i in range(num_colors):
        img[(img >= i * 255 / num_colors) & (img < (i + 1) * 255 / num_colors)] = i * 255 / (num_colors - 1)
    return img


def remove_all_but(img, color, threshold=30):
    distances = np.linalg.norm(img - color, axis=2)
    mask = distances <= threshold
    img[mask] = [0, 0, 0]
    img[~mask] = [255, 255, 255]
    return img


def slice_image(img, roi_x=1215, roi_y=384, roi_width=1078, roi_height=177):
    img_copy = img.copy()

    print("Slicing image...")
    num_slice = 24  # Hours per day
    max_y = 60  # Units of minutes
    scale_amount = 4

    img = darken_non_white(img)

    img = reduce_color_count(img, 2)

    img = scale_up(img, scale_amount)
    img_copy = scale_up(img_copy, scale_amount)

    roi_x = roi_x * scale_amount
    roi_y = roi_y * scale_amount
    roi_height = roi_height * scale_amount
    roi_width = roi_width * scale_amount

    row = []

    slice_width_float = int(roi_width / num_slice)

    all_times = []  # Holder for all hours over the day

    for slice_index in range(0, num_slice):
        # Slice of image, corresponds to time bars
        slice_x = roi_x + int(slice_index * slice_width_float)
        slice_of_image = img[roi_y:roi_y + roi_height, slice_x:int(roi_x + (slice_index + 1) * slice_width_float)]

        cv2.rectangle(img_copy, (slice_x, roi_y),
                      (int(roi_x + (slice_index + 1) * slice_width_float), roi_y + roi_height),
                      (0, 255, 0), 2)

        if WANT_DEBUG_SLICE:
            show_until_destroyed('Slice of image', slice_of_image)

        # Slice down the middle
        true_slice = slice_of_image[:, int(slice_width_float / 2), :]
        rows = len(true_slice)

        lower_grid_buffer = 2
        counter = 0
        for y_coord in range(rows):
            if np.sum(true_slice[y_coord]) == 0:
                counter = counter + 1
            if is_close(true_slice[y_coord], [255, 255, 255], 2) and y_coord < rows - lower_grid_buffer:
                counter = 0

        if VERBOSE:
            print(str(slice_index) + ", " + str((max_y * counter / rows)))

        usage_at_time = np.ceil(max_y * counter / rows)

        row.append(usage_at_time)
        all_times.append(usage_at_time)

    if WANT_DEBUG_SLICE:
        cv2.imshow('Grid ROI', img)
        cv2.waitKey(0)

    row.append(np.sum(all_times))

    print("Slice complete, returning")

    return row, img_copy, scale_amount


def scale_up(img, scale_amount):
    width = int(img.shape[1] * scale_amount)
    height = int(img.shape[0] * scale_amount)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def snap_to_grid(img, x, y, w, h):
    buffer = 40
    maximum_offset = 20

    #  Inch up until you find the grid...
    line_row = None
    line_col = None

    # Inch down until you find the grid
    moving_index = 0
    while line_row is None and moving_index < maximum_offset:
        print("Snapping to grid by looking down...")
        # Start buffer above the current point and scoot down
        line_row = extract_line_snap_to_grid(img,
                                             x,
                                             x + buffer,
                                             y - buffer + moving_index,
                                             y + moving_index, "horizontal")
        moving_index = moving_index + 1

    if line_row is None:
        print("Returning error state")
        return error_state

    upper_left_y = y + line_row + moving_index - buffer

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None and moving_index < maximum_offset:
        # Start buffer to the left of the point
        print("Snapping to grid by looking right...")
        line_col = extract_line_snap_to_grid(img,
                                             x - buffer + moving_index,
                                             x + moving_index,
                                             y,
                                             y + buffer, "vertical")
        moving_index = moving_index + 1
    if line_col is None:
        print("Returning error state")
        return error_state

    upper_left_x = x - buffer + line_col + moving_index

    grid_color = img[upper_left_y,
                 upper_left_x - 1, :]

    test = remove_all_but(img.copy(), grid_color, 120)
    # show_until_destroyed("SHOWING WHAT HAPPENS WHEN WE REMOVE ALL", test)
    line_row = None
    line_col = None

    #  Inch down until you find the grid...
    moving_index = 0
    while line_row is None and moving_index < maximum_offset:
        # Look at the gap between the second to last and last hour of the day
        print("Snapping to grid by looking down... (bottom right)")
        line_row = extract_line_snap_to_grid(test,
                                             x + int(23 * w / 24 - buffer / 2),
                                             x + int(23 * w / 24 + buffer / 2),
                                             y + h + moving_index - buffer,
                                             y + h + moving_index,
                                             "horizontal",
                                             grid_color)

        cv2.rectangle(test, (x + int(23 * w / 24 - buffer / 2), y + h + moving_index - buffer),
                      (x + int(23 * w / 24 + buffer / 2), y + h + moving_index), (0, 255, 0), 2)
        # show_until_destroyed("test",test)

        moving_index = moving_index + 1
    if line_row is None:
        print("Returning error state")
        return error_state

    lower_right_y = y + h - buffer + line_row + moving_index

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None and moving_index < maximum_offset:
        print("Snapping to grid by looking right... (right)")

        line_col = extract_line_snap_to_grid(test,
                                             x + w + moving_index - buffer,
                                             x + w + moving_index,
                                             y + h - buffer,
                                             y + h,
                                             "vertical",
                                             grid_color)
        moving_index = moving_index + 1
    if line_col is None:
        print("Returning error state")
        return error_state

    lower_right_x = x + w - buffer + line_col + moving_index

    print("Returning found values")
    return upper_left_x, upper_left_y, lower_right_x - upper_left_x, lower_right_y - upper_left_y


def extract_line_snap_to_grid(img, x0, x1, y0, y1, mode, grid_color=None):
    sub_image = img[y0:y1, x0:x1].copy()

    is_battery = False
    if grid_color is not None and is_battery:
        pixel_value = grid_color
        sub_image = remove_all_but(sub_image, pixel_value, 100)
        pixel_value = [0, 0, 0]
    else:
        sub_image = reduce_color_count(sub_image, 2)
        pixel_value = get_pixel(sub_image, -2)

        if pixel_value is None:
            return None

    if WANT_DEBUG_SUBIMAGE_SNAP or grid_color is not None:
        cv2.imshow('img', sub_image)
        cv2.waitKey(0)

    count_color = pixel_value

    if mode == "horizontal":
        shape = np.shape(sub_image)

        for i in range(shape[0]):
            row_score = 0
            for j in range(shape[1]):
                pixel = sub_image[i, j]
                if is_close(pixel, count_color):
                    row_score = row_score + 1
            if row_score > 0.7 * shape[1]:  # Threshold set by inspection; can be modified
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


def clean_date_string(date_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', date_string)
    return cleaned_string


def get_text(img, roi_x, roi_y, roi_width, roi_height):
    text_y_start = roi_y + int(roi_height * 1.23)
    text_y_end = roi_y + int(roi_height * 1.46)
    text_x_width = int(roi_width / 8)
    first_location = img[text_y_start:text_y_end, roi_x:(roi_x + text_x_width)]
    second_location = img[text_y_start:text_y_end,
                      roi_x + int(roi_width / 2):(roi_x + int(roi_width / 2) + text_x_width)]

    if WANT_DEBUG_TEXT:
        cv2.imshow("First text location", first_location)
        cv2.waitKey(0)

    first_date = clean_date_string(extract_date(first_location).strip())
    second_date = clean_date_string(extract_date(second_location).strip())

    if is_date(second_date):
        is_pm = True
        first_date = get_day_before(second_date)
    else:
        is_pm = False
    return first_date, second_date, is_pm


def extract_date(image):
    text = pytesseract.image_to_string(image)
    return text


def is_date(s):
    try:
        datetime.datetime.strptime(s, '%b %d')
        return True
    except ValueError:
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


def get_day_before(s):
    try:
        dt = datetime.datetime.strptime(s, '%b %d')
        day_before = dt - datetime.timedelta(days=1)
        return day_before.strftime('%b %d')
    except ValueError:
        return None
