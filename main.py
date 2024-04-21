import os
from glob import iglob
import pandas as pd

from utils import *

WANT_DEBUG_LINE_FIND = False
WANT_DEBUG_GRID = True
WANT_DEBUG_SUBIMAGE = True
WANT_DEBUG_TITLE = False
WANT_DEBUG_SLICE = False
WANT_DEBUG_LEFT = False
VERBOSE = True


def find_right_anchor(d, img, img_copy):
    found_flag = False
    n_boxes = len(d['level'])
    upper_right_x = -1
    upper_right_y = -1
    buffer = 25
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
                while line_row is None:
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
                while line_col is None:
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
    buffer = 40
    key_list = ["2A", "12", "AM"]

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if any(key in d['text'][i] for key in key_list):
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
                while line_row is None:
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
                while line_col is None:
                    line_col = extract_line(img,
                                            x - moving_index - buffer,
                                            x - moving_index,
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

        if WANT_DEBUG_TITLE:
            show_until_destroyed("Title", app_extract)

    title = title.lstrip()

    return title


def process_screen_time(filename):
    img = cv2.imread(filename)
    img_copy = img.copy()

    # Darken to ensure the OCR finds the text anchors
    img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)

    title = find_screenshot_title(img)

    img_chunk_num = 3  # Chunk of image to use when searching for anchors
    img_width = np.shape(img)[1]
    img_height = np.shape(img)[0]
    top_removal = int(img_height * 0.15)
    img_left = img[:, 0:int(img_width / img_chunk_num)]
    img_right = img[:, -int(img_width / img_chunk_num):]

    img_left[0:top_removal, :] = get_pixel(img_left, 1)
    img_right[0:top_removal, :] = get_pixel(img_right, 1)

    if WANT_DEBUG_LEFT:
        cv2.imshow('Grid ROI', img_left)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()  # destroys the window showing image

    # Search for left and right anchors
    d_left = pytesseract.image_to_data(img_left, config='--psm 12', output_type=Output.DICT)
    d_right = pytesseract.image_to_data(img_right, config='--psm 12', output_type=Output.DICT)

    # Correct for offset in right chunk of image
    right_offset = img_width - int(img_width / img_chunk_num)

    for i in range(len(d_right['level'])):
        d_right['left'][i] = d_right['left'][i] + right_offset

    # Find anchors
    found_12, lower_left_x, lower_left_y = find_left_anchor(d_left, img, img_copy)
    found_60, upper_right_x, upper_right_y = find_right_anchor(d_right, img, img_copy)

    if found_60 and found_12:
        roi_x = lower_left_x
        roi_y = upper_right_y
        roi_width = upper_right_x - lower_left_x
        roi_height = lower_left_y - upper_right_y
        # roi_x, roi_y, roi_width, roi_height = snap_to_grid(img, roi_x, roi_y, roi_width, roi_height)

        return load_image(filename, title, roi_x, roi_y, roi_width, roi_height)
    else:
        print("Couldn't find graph anchors!")
        show_until_destroyed('Image', img_copy)


def scale(img, scale_amount):
    width = int(img.shape[1] * scale_amount)
    height = int(img.shape[0] * scale_amount)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def load_image(name, title, roi_x=1215, roi_y=384, roi_width=1078, roi_height=177):
    img = cv2.imread(name)
    num_slice = 24  # Hours per day
    max_y = 60  # Units of minutes

    scale_amount = 4

    img = scale(img, scale_amount)
    img_copy = img.copy()
    x_correction_factor = 6
    y_correction_factor = 8
    height_correction_factor = -4

    roi_x = roi_x * scale_amount + x_correction_factor
    roi_y = roi_y * scale_amount + y_correction_factor
    roi_height = roi_height * scale_amount + height_correction_factor
    roi_width = roi_width * scale_amount

    slice_width = int(roi_width / num_slice)

    row = [title]

    if WANT_DEBUG_GRID:
        cv2.imshow('Grid ROI', img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    print("Saving image to debug: ")
    save_name = f"debug/{name.split('/')[-1]}"
    print(save_name)
    cv2.imwrite(save_name, img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width])

    all_times = []  # Holder for all hours over the day

    for slice_index in range(0, num_slice):
        # Slice of image, corresponds to time bars
        slice_x = roi_x + slice_index * slice_width
        slice_of_image = img[roi_y:roi_y + roi_height, slice_x:slice_x + slice_width]
        slice_of_image = remove_line_color(slice_of_image)
        slice_of_image = darken_non_white(slice_of_image)
        slice_of_image = reduce_color_count(slice_of_image, 2)

        if WANT_DEBUG_SLICE:
            show_until_destroyed('Slice of image', slice_of_image)
        cv2.rectangle(img_copy, (slice_x, roi_y), (slice_x + slice_width, roi_y + roi_height), (0, 255, 0), 2)

        # Slice down the middle
        off_white_threshold = 250 * 3
        true_slice = slice_of_image[:, int(slice_width / 2), :]

        rows = len(true_slice)
        counter = 0
        for y_coord in range(rows - 2):
            if true_slice[y_coord][0] == true_slice[y_coord + 2][0] and np.sum(
                    true_slice[y_coord]) < off_white_threshold:
                counter = counter + 1
        if true_slice[rows - 2][2] and np.sum(true_slice[y_coord]) < off_white_threshold:
            counter = counter + 1
        if true_slice[rows - 1][2] and np.sum(true_slice[y_coord]) < off_white_threshold:
            counter = counter + 1
        if VERBOSE:
            print(str(slice_index) + ", " + str((max_y * counter / rows)))

        usage_at_time = np.ceil(max_y * counter / rows)

        row.append(usage_at_time)
        all_times.append(usage_at_time)

    cv2.imwrite(f"debug/slice_{name.split('/')[-1]}", img_copy)

    row.append(np.sum(all_times))
    return row


if __name__ == '__main__':
    process_screen_time("data/jenny_ipad/IMG_0176 Cropped.png")
    # process_screen_time("../../Downloads/Screenshot 2023-05-11 at 2.10.27 PM.png")
    acceptable_types = ["PNG", "JPG"]  # Only run these files
    omit_keys = ["Parental"]  # Don't run files with these
    root_directory = 'bcm/*/'
    df = pd.DataFrame()

    folder_list = [f for f in iglob(root_directory, recursive=False) if os.path.isdir(f)]

    # Loop over all folders in folder list
    for folder in folder_list:
        all_rows = []
        participant = folder.split("/")[-2]
        print(participant)

        file_list = [f for f in iglob(folder + "**/*", recursive=True) if
                     os.path.isfile(f) and (f[-3:] in acceptable_types)]

        # Recursively loop over all files
        for file in file_list:
            print("Running " + file + "...")
            if not any(key in file for key in omit_keys):
                row = process_screen_time(file)

                if row is not None:
                    # Data-specific date formats
                    day = file.split("/")[2].split(" ")[0] + " " + file.split("/")[2].split(" ")[1]
                    date = file.split("/")[2].split(" ")[2]
                    row = [file, day, date] + row
                    all_rows.append(row)

        # If data extraction successful...
        if len(all_rows) > 0:
            df = pd.DataFrame(np.squeeze(all_rows),
                              columns=['Filename', 'Day', 'Date', 'Title'] + list(range(24)) + ["Total"])
            sorted_df = df.sort_values(by=['Filename'], ascending=True)

            with pd.ExcelWriter('output/Screen Time ' + participant + '.xlsx') as writer:
                sorted_df.to_excel(writer, sheet_name='sheet1')
