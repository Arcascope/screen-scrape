# Steps we need to follow:
# 1. Load image. If dark mode, invert.
# 2. Collect clicks from users
# 3. Snap to grid from clicks
# 4. Using grid location, identify date.
# 5. Extract timing info from graph.
# 6. Store info with date.
# 7. Save on every iteration so it cannot be lost accidentally.


import os
from glob import iglob
import pandas as pd

from utils import *


def snap_to_grid(img, x, y, w, h):
    buffer = 40

    #  Inch up until you find the grid...
    line_row = None
    line_col = None

    # Inch down until you find the grid
    moving_index = 0
    while line_row is None:
        line_row = extract_line(img,
                                x,
                                x + buffer,
                                y - buffer + moving_index,
                                y + moving_index, "horizontal")
        moving_index = moving_index + 1
    upper_left_y = y + line_row + moving_index - buffer

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None:
        line_col = extract_line(img,
                                x - buffer + moving_index,
                                x + moving_index,
                                y,
                                y + buffer, "vertical")
        moving_index = moving_index + 1
    upper_left_x = x - buffer + line_col + moving_index

    line_row = None
    line_col = None

    #  Inch up until you find the grid...
    moving_index = 0
    while line_row is None:
        line_row = extract_line(img,
                                x + w - buffer,
                                x + w + buffer,
                                y + h + moving_index - buffer,
                                y + h + moving_index,
                                "horizontal")

        moving_index = moving_index + 1
    lower_right_y = y + h - buffer + line_row + moving_index

    #  Inch right until you find the grid...
    moving_index = 0
    while line_col is None:
        line_col = extract_line(img,
                                x + w + moving_index - buffer,
                                x + w + moving_index,
                                y + h - buffer,
                                y + h + buffer,
                                "vertical")
        moving_index = moving_index + 1
    lower_right_x = x + w - buffer + line_col + moving_index

    print(x, y, w, h)
    print(upper_left_x, upper_left_y, lower_right_x - upper_left_x, lower_right_y - upper_left_y)
    return upper_left_x, upper_left_y, lower_right_x - upper_left_x, lower_right_y - upper_left_y


def process_battery(filename):
    img = cv2.imread(filename)
    img = scale_up(img, 4)
    get_clicks(filename, img)


def get_text_rect(img, roi_x, roi_y, roi_width, roi_height):
    text_y_start = roi_y + int(roi_height * 1.23)
    text_y_end = roi_y + int(roi_height * 1.46)
    text_x_width = int(roi_width / 8)
    first_location = img[text_y_start:text_y_end, roi_x:(roi_x + text_x_width)]
    cv2.imshow("first loca", first_location)
    cv2.waitKey(0)
    print(extract_date(first_location))


def get_clicks(name, img):
    global detecting_clicks
    global click_points
    detecting_clicks = True
    click_points = []
    cv2.namedWindow(winname=name)

    cv2.imshow(name, img)
    cv2.setMouseCallback(name, store_click)
    while detecting_clicks:
        cv2.waitKey(1)

    roi_x = click_points[0][0]
    roi_y = click_points[0][1]
    roi_width = click_points[1][0] - roi_x
    roi_height = click_points[1][1] - roi_y
    cv2.destroyAllWindows()

    img_copy = img.copy()

    roi_x, roi_y, roi_width, roi_height = snap_to_grid(img, roi_x, roi_y, roi_width, roi_height)
    row = slice_image(img_copy, name, roi_x, roi_y, roi_width, roi_height)
    text_rect = get_text_rect(img_copy, roi_x, roi_y, roi_width, roi_height)

    roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    cv2.imshow(name, roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def slice_image(img, title, roi_x=1215, roi_y=384, roi_width=1078, roi_height=177):
    num_slice = 24  # Hours per day
    max_y = 60  # Units of minutes
    dark_blue = [255, 121, 0]
    scale_amount = 1
    img = scale_up(img, scale_amount)
    roi_x = roi_x * scale_amount
    roi_y = roi_y * scale_amount
    roi_height = roi_height * scale_amount
    roi_width = roi_width * scale_amount

    row = [title]
    roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    slice_width_float = int(roi_width / num_slice)

    if WANT_DEBUG_GRID:
        cv2.imshow('Grid ROI', roi)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()  # destroys the window showing image
    if VERBOSE:
        print("Does the image look right?")

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
        off_white_threshold = 250 * 3
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
    process_battery("data/Example-Battery-Image.png")
    root_directory = 'data/*/'
    # df = pd.DataFrame()
    #
    folder_list = [f for f in iglob(root_directory, recursive=False) if os.path.isdir(f)]

    # Loop over all folders in folder list
    for folder in folder_list:
        all_rows = []
        participant = folder.split("/")[-2]
        print(participant)

        file_list = [f for f in iglob(folder + "**/*", recursive=True) if
                     os.path.isfile(f)]

        # Recursively loop over all files
        for file in file_list:
            print("Running " + file + "...")
            row = process_battery(file)

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
