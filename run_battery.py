import datetime
import os
from glob import iglob
import pandas as pd

from utils import *

error_state = -1, -1, -1, -1

def is_date(s):
    try:
        datetime.datetime.strptime(s, '%b %d')
        return True
    except ValueError:
        return False

def get_day_before(s):
    try:
        dt = datetime.datetime.strptime(s, '%b %d')
        day_before = dt - datetime.timedelta(days=1)
        return day_before.strftime('%b %d')
    except ValueError:
        return None

def snap_to_grid(img, x, y, w, h):
    buffer = 40
    maximum_offset = 20

    #  Inch up until you find the grid...
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


def process_battery(filename):
    img = cv2.imread(filename)
    img = scale_up(img, 4)
    return get_clicks(filename,img)


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

    first_date = extract_date(first_location).strip()
    second_date = extract_date(second_location).strip()

    if is_date(second_date):
        is_pm = True
        first_date = get_day_before(second_date)
    else:
        is_pm = False
    return first_date, second_date, is_pm


def get_clicks(name,img):
    is_valid = False
    msg = "Please select upper left and lower right corners"
    rows = []
    while not is_valid:
        global detecting_clicks
        global click_points
        detecting_clicks = True
        click_points = []

        scale_factor = 0.1
        cv2.namedWindow(msg, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(msg, img)
        cv2.resizeWindow(msg, round(scale_factor * img.shape[1]), round(scale_factor * img.shape[0]))


        cv2.setMouseCallback(msg, store_click)
        while detecting_clicks:
            cv2.waitKey(1)

        roi_x = click_points[0][0]
        roi_y = click_points[0][1]
        roi_width = click_points[1][0] - roi_x
        roi_height = click_points[1][1] - roi_y
        cv2.destroyAllWindows()

        if roi_width <= 0 or roi_height <= 0:
            msg = "Invalid clicks! Please try again"
            print(msg)

            is_valid = False
            continue

        roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        save_name = name.split('\\')[-1].split('.')[0]
        cv2.imwrite(f"./debug/{save_name}_clicked.png", roi)

        expl = "Clicked region (Press space to continue)"
        cv2.namedWindow(expl, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(expl, roi)
        cv2.resizeWindow(expl, round(scale_factor * roi.shape[1]), round(scale_factor * roi.shape[0]))
        cv2.waitKey(0)

        img_copy = img.copy()

        roi_x, roi_y, roi_width, roi_height = snap_to_grid(img, roi_x, roi_y, roi_width, roi_height)
        roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        if roi_width <= 0 or roi_height <= 0:
            msg = "Grid detection failed! Please try again"
            print(msg)
            is_valid = False
            continue

        if WANT_DEBUG_GRID:
            cv2.imwrite(f"debug/{save_name}_updated.png", roi)

            grid = 'Updated grid (Press space to continue)'
            cv2.namedWindow(grid, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(grid, roi)
            cv2.resizeWindow(grid, round(scale_factor * roi.shape[1]), round(scale_factor * roi.shape[0]))

            cv2.waitKey(0)
            cv2.destroyAllWindows()  # destroys the window showing image

        answer = input("Did the grid look right? (y/n)")

        if answer == "n" or answer == "N":
            is_valid = False
            continue

        is_valid = True
        row_raw = slice_image(img_copy, roi_x, roi_y, roi_width, roi_height)
        text1, text2, is_pm = get_text(img_copy, roi_x, roi_y, roi_width, roi_height)
        if is_pm:
            row1 = [text1] + [-1] * 12 + row_raw[:12] + [-1]
            row2 = [text2] + row_raw[12:24] + [-1] * 12 + [-1]
            rows.append(row1)
            rows.append(row2)

        else:
            row = [text1] + row_raw
            rows.append(row)

    return rows


def scale_up(img, scale_amount):
    width = int(img.shape[1] * scale_amount)
    height = int(img.shape[0] * scale_amount)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def remove_all_but(img, color, threshold=30):
    shape = np.shape(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = img[i, j]

            if is_close(pixel, color, threshold):
                img[i, j] = [0, 0, 0]
            else:
                img[i, j] = [255, 255, 255]

    return img


def slice_image(img, roi_x=1215, roi_y=384, roi_width=1078, roi_height=177):
    print("Slicing image...")
    num_slice = 24  # Hours per day
    max_y = 60  # Units of minutes
    dark_blue = [255, 121, 0]
    scale_amount = 1
    img = scale_up(img, scale_amount)
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
        slice_of_image = remove_all_but(slice_of_image, dark_blue)
        cv2.rectangle(img, (slice_x, roi_y),
                      (int(roi_x + (slice_index + 1) * slice_width_float), roi_y + roi_height),
                      (0, 255, 0), 2)

        if WANT_DEBUG_SLICE:
            show_until_destroyed('Slice of image', slice_of_image)

        # Slice down the middle
        true_slice = slice_of_image[:, int(slice_width_float / 2), :]
        rows = len(true_slice)
        counter = 0
        for y_coord in range(rows):
            if np.sum(true_slice[y_coord]) == 0:
                counter = counter + 1
        if VERBOSE:
            print(str(slice_index) + ", " + str((max_y * counter / rows)))

        usage_at_time = np.ceil(max_y * counter / rows)

        row.append(usage_at_time)
        all_times.append(usage_at_time)

    if WANT_DEBUG_SLICE:
        cv2.imshow('Grid ROI', img)
        cv2.waitKey(0)

    row.append(np.sum(all_times))
    return row


def store_click(event, x, y, flags, param):
    global detecting_clicks
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Click detected at {x},{y}')
        click_points.append([x, y])
        if len(click_points) == 2:
            detecting_clicks = False


if __name__ == '__main__':
    # rows = process_battery("data/usc-data-data/1174/1174_10.30.20_11.25.jpg") # AM/PM test canonical example
    # process_battery("data/usc-data-data/1174/1174_10.16.20_21.02.jpg")  # Dark mode canonical example
    # process_battery("data/Example-Battery-Image.png")
    root_directory = 'data/usc-data/*/'
    folder_list = [f for f in iglob(root_directory, recursive=False) if os.path.isdir(f)]

    # Loop over all folders in folder list
    for folder in folder_list:
        all_rows = []
        participant = folder.split("/")[-2]

        print(f"For {participant}...")

        file_list = [f for f in iglob(folder + "**/*", recursive=True) if
                     os.path.isfile(f)]

        # Recursively loop over all files
        for file in file_list:
            print("Running " + file + "...")
            rows = process_battery(file)

            if rows is not None:
                # Data-specific date formats
                print(file)
                date = file.split("/")[-1].split("_")[1]
                for row in rows:
                    row = [file, date] + row
                    all_rows.append(row)

            print(all_rows)
            # If data extraction successful...
            if len(all_rows) > 0:
                df = pd.DataFrame(all_rows,
                                  columns=['Filename', 'File date', 'Date from Image'] + list(range(24)) + ["Total"])
                sorted_df = df.sort_values(by=['Filename'], ascending=True)

                with pd.ExcelWriter('output/Battery ' + participant + '.xlsx') as writer:
                    sorted_df.to_excel(writer, sheet_name='sheet1')
