import glob
import json
import os
import sys

import gatetools as gt
import itk
import numpy as np
import pydicom
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from auto_lattice_planner import lattice_planner_helper


def select_CT_folder():
    folder = QFileDialog.getExistingDirectory(None, "Select CT Folder")
    return folder


def select_RTStruct_file():
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select RTStruct DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
    )
    return file_path


def OAR_mask_merger(masks):

    for mask, margin in masks:
        mask = mask.astype(bool)


class ROIPicker(QWidget):
    def __init__(self, rtstruct_path):
        super().__init__()

        self.user_cancelled = False  # <--- flag for cancellation

        self.setWindowTitle("Select Lattice Settings")
        self.resize(400, 600)

        self.selected_roi = None
        self.lattice_geometry = "Normal (Duriseti)"  # default

        self.rtstruct = pydicom.dcmread(rtstruct_path)
        self.roi_names = [roi.ROIName for roi in self.rtstruct.StructureSetROISequence]

        # --- Geometry selection ---
        self.radio_label = QLabel("Select Lattice Geometry")

        self.radio1 = QRadioButton("Normal (Duriseti)")
        self.radio2 = QRadioButton("Large (+33%)")

        self.radio1.setChecked(True)

        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.radio1)
        self.radio_group.addButton(self.radio2)

        self.radio_group.buttonClicked.connect(self.radio_selected)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio1)
        radio_layout.addWidget(self.radio2)

        # Tolerance Slider
        self.slider_label = QLabel("Overlap Tolerance")
        self.slider = QSlider(Qt.Horizontal)

        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slider_value_changed)

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(
            QLabel("Set tolerance (1.0 = full sphere inside volume, 0.1 = 10% inside)")
        )
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)

        self.slider_label_2 = QLabel("1")  # or whatever default

        self.slider_value_changed(self.slider.value())

        # --- ROI list ---
        self.label = QLabel("Select ROI from the list:")
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.roi_names)

        # --- Buttons ---
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        self.skip_button = QPushButton("Cancel")
        self.skip_button.clicked.connect(self.cancel_selection)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.skip_button)

        # --- Main layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.radio_label)
        layout.addLayout(radio_layout)
        layout.addWidget(self.slider_label)
        layout.addLayout(slider_layout)
        layout.addWidget(self.slider_label_2)
        layout.addWidget(self.label)
        layout.addWidget(self.list_widget)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def radio_selected(self, button):
        self.lattice_geometry = button.text()
        print(f"Geometry Size: {self.lattice_geometry}")

    def slider_value_changed(self, value):
        scale = 10
        self.slider_label_2.setText("{}".format(value / scale))

        self.tolerance_value = value / scale

    def confirm_selection(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an ROI.")
            return

        self.selected_roi = selected_items[0].text()
        self.close()

    def cancel_selection(self):
        self.selected_roi = None
        self.user_cancelled = True
        self.close()

    def closeEvent(self, event):
        if event.spontaneous():
            # User clicked the X or Alt+F4
            self.user_cancelled = True
            print("Window closed by user (X)")
        else:
            # Closed programmatically via self.close()
            print("Window closed programmatically")
        event.accept()


class ROIPicker2(QWidget):
    def __init__(self, rtstruct_path):
        super().__init__()
        self.resize(500, 600)
        self.user_cancelled = False

        self.setWindowTitle("Select Target ROI")
        self.selected_roi = None
        self.rtstruct = pydicom.dcmread(rtstruct_path)
        self.roi_names = [roi.ROIName for roi in self.rtstruct.StructureSetROISequence]

        self.layout = QVBoxLayout()
        self.label = QLabel("Select ROI from the list:")
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.roi_names)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_selection)

        # Arrange buttons horizontally
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.skip_button)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.list_widget)
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def confirm_selection(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an ROI.")
            return
        self.selected_roi = selected_items[0].text()
        self.close()

    def skip_selection(self):
        self.selected_roi = None
        self.close()

    def closeEvent(self, event):
        if event.spontaneous():
            # User clicked the X or Alt+F4
            self.user_cancelled = True
            print("Window closed by user (X)")
        else:
            # Closed programmatically via self.close()
            print("Window closed programmatically")
        event.accept()


class OARRowWidget(QWidget):
    def __init__(self, roi_name):
        super().__init__()

        self.checkbox = QCheckBox(roi_name)

        self.margin_box = QComboBox()
        self.margin_box.addItems(["10", "15"])
        self.margin_box.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addStretch()
        layout.addWidget(QLabel("Margin (mm):"))
        layout.addWidget(self.margin_box)
        layout.setContentsMargins(5, 2, 5, 2)
        self.setLayout(layout)

        self.checkbox.stateChanged.connect(self._toggle_margin)

    def _toggle_margin(self, state):
        self.margin_box.setEnabled(state == Qt.Checked)

    def is_checked(self):
        return self.checkbox.isChecked()

    def roi_name(self):
        return self.checkbox.text()

    def margin(self):
        return float(self.margin_box.currentText())


class OARPicker(QWidget):
    def __init__(self, rtstruct_path):
        super().__init__()

        self.user_cancelled = False

        self.setWindowTitle("Select OARs to avoid")
        self.resize(500, 600)

        # Final output: {roi_name: margin_mm}
        self.selected_oars = {}

        self.rtstruct = pydicom.dcmread(rtstruct_path)
        self.roi_names = [roi.ROIName for roi in self.rtstruct.StructureSetROISequence]

        self.layout = QVBoxLayout()
        self.label = QLabel("Select OARs and assign protective margins:")
        self.list_widget = QListWidget()

        self.oar_rows = []

        for roi in self.roi_names:
            item = QListWidgetItem(self.list_widget)
            row_widget = OARRowWidget(roi)

            item.setSizeHint(row_widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, row_widget)

            self.oar_rows.append(row_widget)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_oar_selection)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_oar_selection)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.skip_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.list_widget)
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def confirm_oar_selection(self):
        selected = []

        for row in self.oar_rows:
            if row.is_checked():
                selected.append((row.roi_name(), row.margin()))

        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one OAR.")
            return

        self.selected_oars = selected
        self.close()

    def skip_oar_selection(self):
        self.selected_oars = {}
        self.close()

    def closeEvent(self, event):
        if event.spontaneous():
            # User clicked the X or Alt+F4
            self.user_cancelled = True
            print("Window closed by user (X)")
        else:
            # Closed programmatically via self.close()
            print("Window closed programmatically")
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Step 1: Select CT folder
    dicom_series_path = select_CT_folder()

    if not dicom_series_path:
        print("No CT folder selected.")
        sys.exit(0)

    dicom_series_path = os.path.normpath(dicom_series_path)

    # Step 2: Select RTStruct file
    rtstruct_path = select_RTStruct_file()

    if not rtstruct_path:
        print("No RTStruct selected.")
        sys.exit(0)

    # Step 3: Pick primary ROI (main target)
    picker = ROIPicker(rtstruct_path)
    picker.show()
    app.exec_()

    # while picker.isVisible():
    #     app.processEvents()  # <-- process Qt events manually

    if picker.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    # Step 4: Optional - Pick secondary ROI
    picker2 = ROIPicker2(rtstruct_path)
    picker2.setWindowTitle("Select Optional Additional Target ROI")
    picker2.show()
    app.exec_()

    if picker2.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    # Step 5: Select OARs to be given additional margin
    picker3 = OARPicker(rtstruct_path)
    picker3.setWindowTitle("Select OARs and required margin")
    picker3.show()
    app.exec_()

    if picker3.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    primary_roi = picker.selected_roi
    lattice_geometry = picker.lattice_geometry
    tolerance = picker.tolerance_value
    print(f"Primary ROI selected: {primary_roi}")

    secondary_roi = picker2.selected_roi

    additional_target = False
    if secondary_roi is not None:
        print(f"Additional ROI selected: {secondary_roi}")
        additional_target = True
    else:
        print("No additional ROI selected.")

    selected_oars = picker3.selected_oars
    print("OARs to avoid:", selected_oars)

    output_folder = "."

    # Create the lattice data object
    data = lattice_planner_helper.LatticeData(
        dicom_series_path=dicom_series_path,
        rtstruct_path=rtstruct_path,
        additional_target=additional_target,
        lattice_geometry=lattice_geometry,
        tolerance=tolerance,
        primary_roi=primary_roi,
        secondary_roi=secondary_roi,
        selected_oars=selected_oars,
        output_folder=output_folder,
    )

    with open("paths.json", "w") as f:
        json.dump(data.to_json(), f, indent=4)

    lattice_planner_helper.compute_volumes(data)


if __name__ == "__main__":

    main()
