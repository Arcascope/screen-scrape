import sys
import os
import re

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QComboBox, QFileDialog, QHBoxLayout, QMessageBox, QFrame, QCheckBox,
                             QDesktopWidget)
from PyQt5.QtGui import QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QLineEdit

from image_processor import process_image, process_image_with_grid
from magnifying_label import MagnifyingLabel
import pandas as pd
import pytesseract


# Note: May need to be imported on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class BatteryRow:
    @classmethod
    def headers(cls):
        return ['Full path', 'File name', 'ID',
                'Date from file name', "Date from image",
                "Time from image"] + list(range(24)) + ["Total"]

    def __init__(self, full_path, file_name, date_from_image, time_from_ui, rows):
        super().__init__()
        self.full_path = full_path
        self.file_name = file_name
        self.date_from_image = date_from_image
        self.time_from_ui = time_from_ui
        self.rows = rows

    def date_extracted_from_file_name(self):
        match = re.search(r"\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4}", self.full_path)

        if match:
            return match.group(0)
        return ""

    def subject_id_extracted_from_file_path(self):
        directory = os.path.dirname(self.full_path)
        return os.path.basename(directory)

    def to_csv_row(self):
        return [self.full_path, self.file_name, self.subject_id_extracted_from_file_path(),
                self.date_extracted_from_file_name(), self.date_from_image,
                self.time_from_ui] + self.rows


class ScreenTimeRow:
    @classmethod
    def headers(cls):
        return ['Full path', 'File name', 'ID', 'Date from file name', "Date from image", "App title"] + list(
            range(24)) + ["Total"]

    def __init__(self, full_path, file_name, date_from_image, app_title, rows):
        super().__init__()
        self.full_path = full_path
        self.file_name = file_name
        self.app_title = app_title
        self.date_from_image = date_from_image
        self.rows = rows

    def date_extracted_from_file_name(self):
        match = re.search(r"\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4}", self.full_path)

        if match:
            return match.group(0)
        return ""

    def subject_id_extracted_from_file_path(self):
        path_parts = self.full_path.split(os.sep)
        if len(path_parts) > 3:
            return path_parts[-3]
        else:
            return ""

    def to_csv_row(self):
        return [self.full_path, self.file_name, self.subject_id_extracted_from_file_path(),
                self.date_extracted_from_file_name(), self.date_from_image, self.app_title] + self.rows


class ScreenshotApp(QWidget):
    length_dimension = 600

    def __init__(self):
        super().__init__()
        self.time_mapping = None
        self.date_from_image_label = None
        self.app_name_label = None

        self.mode = "Battery"
        self.graph_image_label = None
        self.snap_to_grid_checkbox = None
        self.magnifier_label = None
        self.next_button = None
        self.previous_button = None
        self.time_label = None
        self.time_slider = None
        self.date_from_image_label_edit = None
        self.app_name_label_edit = None
        self.image_name_line_edit = None
        self.cropped_image_label = None
        self.image_label = None
        self.instruction_label = None
        self.folder_button = None
        self.images = []
        self.current_image_index = 0
        self.click_count = 0
        self.coordinates = []
        self.folder_name = ""
        self.init_ui()
        self.current_row = None
        self.failed = None

    def init_ui(self):
        self.setWindowTitle('Screenshot Slideshow')
        self.setStyleSheet(
            "QWidget { font-size: 14px; } QPushButton { font-weight: bold; }")
        #
        layout = QVBoxLayout(self)

        btn_battery = QPushButton('Select Folder of Battery Images')
        btn_battery.clicked.connect(lambda: self.open_folder('Battery'))
        btn_screen_time = QPushButton('Select Folder of Screen Time Images')
        btn_screen_time.clicked.connect(lambda: self.open_folder('ScreenTime'))
        layout.addWidget(btn_battery)
        layout.addWidget(btn_screen_time)
        self.instruction_label = QLabel(
            "Click Next/Save if the graphs match, otherwise click the upper left corner of the graph in the left image."
        )
        self.instruction_label.setStyleSheet(
            "background-color:rgb(255,255,150)")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFixedHeight(50)
        layout.addWidget(self.instruction_label)
        self.instruction_label.hide()

        main_image_layout = QHBoxLayout()
        layout.addLayout(main_image_layout)
        original_image_layout = QVBoxLayout()

        self.image_label = MagnifyingLabel(self)
        self.image_label.setCursor(Qt.CrossCursor)

        self.magnifier_label = QLabel(self)
        self.magnifier_label.resize(100, 100)
        self.magnifier_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setAlignment(Qt.AlignTop)

        original_image_layout.addWidget(self.image_label)
        main_image_layout.addLayout(original_image_layout)

        cropped_image_layout = QVBoxLayout()
        self.cropped_image_label = QLabel("No cropped image loaded.")
        self.cropped_image_label.setAlignment(Qt.AlignCenter)
        cropped_image_layout.addWidget(self.cropped_image_label)

        self.graph_image_label = QLabel("No graph extracted.")
        self.graph_image_label.setAlignment(Qt.AlignCenter)
        cropped_image_layout.addWidget(self.graph_image_label)
        cropped_image_layout.setAlignment(Qt.AlignCenter)
        main_image_layout.addLayout(cropped_image_layout)

        text_fields_layout = QVBoxLayout()
        text_fields_layout.setAlignment(Qt.AlignCenter)
        text_fields_layout.setSpacing(5)
        text_fields_layout.addWidget(QLabel("Image Name:"))
        self.image_name_line_edit = QLineEdit("IMG_8383.png")
        text_fields_layout.addWidget(self.image_name_line_edit)
        #
        self.date_from_image_label = QLabel("Date from image:")
        text_fields_layout.addWidget(self.date_from_image_label)
        self.date_from_image_label_edit = QLineEdit("")
        text_fields_layout.addWidget(self.date_from_image_label_edit)

        self.app_name_label = QLabel("App Name:")
        text_fields_layout.addWidget(self.app_name_label)
        self.app_name_label_edit = QLineEdit("")
        text_fields_layout.addWidget(self.app_name_label_edit)

        self.snap_to_grid_checkbox = QCheckBox(
            "Automatically snap to grid", self)
        text_fields_layout.addWidget(self.snap_to_grid_checkbox)

        main_image_layout.addLayout(text_fields_layout)
        main_image_layout.setStretch(0, 1)
        main_image_layout.setStretch(1, 1)
        main_image_layout.setStretch(2, 1)

        self.time_mapping = {0: "Midnight", 1: "3 AM", 2: "6 AM", 3: "9 AM",
                             4: "12 PM", 5: "3 PM", 6: "6 PM", 7: "9 PM"}
        slider_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time_mapping) - 1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        slider_layout.addWidget(self.slider)
        self.slider.hide()
        layout.addLayout(slider_layout)

        self.setLayout(layout)
        self.setMinimumSize(500, 50)

        self.slider.valueChanged.connect(self.update_time_label)

        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.time_label = QLabel(
            "First time displayed in screenshot: Midnight")
        self.time_label.setAlignment(Qt.AlignCenter)
        label_layout.addWidget(self.time_label)
        self.time_label.hide()
        label_layout.addStretch()
        layout.addLayout(label_layout)

        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton('Previous')
        self.previous_button.clicked.connect(self.show_previous_image)
        nav_layout.addWidget(self.previous_button)

        self.next_button = QPushButton('Next/Save')
        self.next_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        screen_geo = QDesktopWidget().screenGeometry()

        self.resize(int(screen_geo.width() * 0.8),
                    int(screen_geo.height() * 0.8))

    def update_instruction_label(self):
        if not self.failed:
            self.instruction_label.setText(
                "Click Next/Save if the graphs match, otherwise click the upper left corner of the graph in the left image to reselect."
            )
            self.instruction_label.setStyleSheet(
                "background-color:rgb(255,255,150)")
        else:
            self.instruction_label.setText(
                "An error occurred. To start reselecting, first click the upper left corner of the graph in the left image."
            )
            self.instruction_label.setStyleSheet(
                "background-color:rgb(255,0,0)")

    def adjust_ui_for_image_type(self, image_type):

        self.app_name_label.setText("Extracted app name:")
        self.app_name_label_edit.show()

        self.date_from_image_label.setText("Date of first displayed time:")
        self.date_from_image_label_edit.show()

        if image_type == "Battery":
            self.slider.show()
            self.time_label.show()
            self.instruction_label.show()

        elif image_type == "ScreenTime":
            self.slider.hide()
            self.time_label.hide()
            self.instruction_label.show()

    def capture_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.coordinates.append((x, y))
        self.click_count += 1

        if self.click_count == 1:
            self.instruction_label.setText(
                "Now click the bottom right corner of the graph in the left image to finish the selection."
            )
            self.instruction_label.setStyleSheet(
                "background-color:rgb(0,255,0)")
        elif self.click_count == 2:
            self.process_coordinates(self.coordinates[0], self.coordinates[1])
            self.coordinates = []
            self.click_count = 0
            self.update_instruction_label()

    def process_coordinates(self, upper_left, lower_right):
        image_path = self.images[self.current_image_index]
        original_pixmap = QPixmap(image_path)

        original_width = original_pixmap.width()
        original_height = original_pixmap.height()

        scalar = self.length_dimension / original_height

        display_width = original_width * scalar
        display_height = original_height * scalar

        scale_x = original_width / display_width
        scale_y = original_height / display_height

        true_upper_left = (
            int(upper_left[0] * scale_x), int(upper_left[1] * scale_y))
        true_lower_right = (
            int(lower_right[0] * scale_x), int(lower_right[1] * scale_y))

        print(
            f"True Upper Left: {true_upper_left}, True Lower Right: {true_lower_right}")
        print(f"Orig width: {original_width}, Orig height: {original_height}")
        print(
            f"Display width: {display_width}, Display height: {display_height}")

        print("Processing image from clicks...")
        processed_image_path, graph_image_path, row, title, date_in_image = process_image_with_grid(image_path,
                                                                                                    true_upper_left,
                                                                                                    true_lower_right,
                                                                                                    self.mode == "Battery",
                                                                                                    self.snap_to_grid_checkbox.isChecked())

        self.update(title, date_in_image, row,
                    processed_image_path, graph_image_path)

    def update_time_label(self, value):
        selected_time = self.time_mapping.get(value, "Midnight")
        self.time_label.setText(
            f"First time displayed in screenshot: {selected_time}")

    def open_folder(self, selection_type):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_name = folder_path
        self.mode = selection_type
        self.adjust_ui_for_image_type(selection_type)
        if folder_path:

            self.load_images(folder_path)
            if self.images:
                self.show_image(0)
            else:
                QMessageBox.information(
                    self, "No Images Found", "No images found in the folder.")

    def load_images(self, folder_path):

        self.images = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')):
                    self.images.append(os.path.join(root, filename))
        if not self.images:
            self.image_label.setText("No image loaded.")
            self.cropped_image_label.setText("No cropped image loaded.")
            self.image_name_line_edit.setText("Image name will appear here.")

    def update(self, title, date_in_image, row, processed_image_path, graph_image_path):

        self.date_from_image_label_edit.setText(date_in_image)
        self.app_name_label_edit.setText(title)
        if self.mode == "Battery":
            self.current_row = BatteryRow(full_path=self.images[self.current_image_index],
                                          file_name=self.image_name_line_edit.text(),
                                          date_from_image=self.date_from_image_label_edit.text(),
                                          time_from_ui=self.time_mapping[self.slider.value(
                                          )],
                                          rows=row)
        else:
            self.current_row = ScreenTimeRow(full_path=self.images[self.current_image_index],
                                             file_name=self.image_name_line_edit.text(),
                                             date_from_image=self.date_from_image_label_edit.text(),
                                             app_title=self.app_name_label_edit.text(),
                                             rows=row)

        if processed_image_path:
            processed_pixmap = QPixmap(processed_image_path)
            self.cropped_image_label.setPixmap(
                processed_pixmap.scaled(self.length_dimension, self.length_dimension, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))
            self.failed = False
            self.update_instruction_label()

        else:
            self.cropped_image_label.setText(
                "No cropped image could be loaded from the selection.")
            self.failed = True
            self.update_instruction_label()

        if graph_image_path:
            processed_graph_pixmap = QPixmap(graph_image_path)
            self.graph_image_label.setPixmap(
                processed_graph_pixmap.scaled(self.length_dimension, self.length_dimension, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
            self.failed = False
            self.update_instruction_label()

        else:
            self.graph_image_label.setText(
                "No graph could be extracted from the selection.")
            self.failed = True
            self.update_instruction_label()

    def show_image(self, index):
        if 0 <= index < len(self.images):
            image_path = self.images[index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(
                pixmap.scaled(self.length_dimension, self.length_dimension, Qt.KeepAspectRatio,
                              Qt.SmoothTransformation))
            self.image_name_line_edit.setText(os.path.basename(image_path))
            self.current_image_index = index

            processed_image_path, graph_image_path, row, title, date_in_image = process_image(image_path,
                                                                                              self.mode == "Battery",
                                                                                              self.snap_to_grid_checkbox.isChecked())

            self.update(title, date_in_image, row,
                        processed_image_path, graph_image_path)

        else:
            self.image_label.setText("No image loaded.")
            self.cropped_image_label.setText("No cropped image loaded.")
            self.graph_image_label.setText("No graph extracted")
            self.image_name_line_edit.setText("No image available.")

    def show_next_image(self):
        if self.current_row and not self.failed:
            self.save_current_row()
            if self.current_image_index + 1 < len(self.images):
                self.show_image(self.current_image_index + 1)
            else:
                self.instruction_label.setText(
                    "Images complete."
                )
                self.instruction_label.setStyleSheet(
                    "background-color:rgb(0,0,0)")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def save_current_row(self):
        if self.current_row:
            csv_path = (
                os.path.dirname(sys.argv[0])
                + "/output/"
                + os.path.basename(self.folder_name)
                + ".csv"
            )

            if self.mode == "Battery":
                headers = BatteryRow.headers()
                self.current_row.date_from_image = self.date_from_image_label_edit.text()
                self.current_row.time_from_ui = self.time_mapping[self.slider.value(
                )]
            else:
                headers = ScreenTimeRow.headers()
                self.current_row.app_title = self.app_name_label_edit.text()
                self.current_row.date_from_image = self.date_from_image_label_edit.text()

            if not os.path.exists(csv_path):
                df = pd.DataFrame(columns=headers)
                df.to_csv(csv_path, index=False)
                print("CSV file created with headers.")

            new_row = pd.DataFrame(
                [self.current_row.to_csv_row()], columns=headers)
            new_row.to_csv(csv_path, mode='a', header=False, index=False)
            print("New row appended to CSV.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ScreenshotApp()
    ex.show()
    sys.exit(app.exec_())
