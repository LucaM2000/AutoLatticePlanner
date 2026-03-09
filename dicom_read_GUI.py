import gatetools as gt
from pathlib import Path
import numpy as np
import glob
import pydicom
import itk
from scipy.ndimage import distance_transform_edt
import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QHBoxLayout, QListWidgetItem, QComboBox, QCheckBox, QRadioButton,QButtonGroup, QSlider)
import json


def select_CT_folder():
    folder = QFileDialog.getExistingDirectory(None, "Select CT Folder")
    return folder


def select_RTStruct_file():
    file_path, _ = QFileDialog.getOpenFileName(None, "Select RTStruct DICOM File", "",
                                               "DICOM Files (*.dcm);;All Files (*)")
    return file_path

def resample_to_reference(input_image, reference_image, interpolator='nearest'):
    """
    Resample input_image to match reference_image geometry (spacing, size, origin, direction)
    """

    # Set up the resampler
    ResampleFilterType = itk.ResampleImageFilter[type(input_image), type(reference_image)]
    resampler = ResampleFilterType.New()

    resampler.SetInput(input_image)
    resampler.SetReferenceImage(reference_image)
    resampler.UseReferenceImageOn()

    # Choose interpolator
    if interpolator == 'nearest':
        InterpolatorType = itk.NearestNeighborInterpolateImageFunction[type(reference_image), itk.D]
        interpolator_obj = InterpolatorType.New()
    else:
        InterpolatorType = itk.LinearInterpolateImageFunction[type(reference_image), itk.D]
        interpolator_obj = InterpolatorType.New()

    resampler.SetInterpolator(interpolator_obj)

    # Execute the resampling
    resampler.Update()
    return resampler.GetOutput()

def OAR_mask_merger(masks):

    for mask, margin in masks:
        mask = mask.astype(bool)


class ROIPicker(QWidget):
    def __init__(self, rtstruct_path):
        super().__init__()

        self.user_cancelled = False  # <--- flag for cancellation

        self.setWindowTitle("Select Lattice Settings")
        self.resize(400, 600)

        self.selected_roi = []
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
        slider_layout.addWidget(QLabel("Set tolerance (1.0 = full sphere inside volume, 0.1 = 10% inside)"))
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)

        self.slider_label_2 = QLabel("1")  # or whatever default

        self.slider_value_changed(self.slider.value())

        # --- ROI list ---
        self.label = QLabel("Select ROI from the list:")
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.roi_names)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)

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

        self.selected_roi = [item.text() for item in selected_items]
        self.close()

    def cancel_selection(self):
        self.selected_roi = []
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
        self.margin_box.addItems(["5", "10", "15"])
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
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one OAR."
            )
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
    ct_files = glob.glob(os.path.join(dicom_series_path, "*.dcm"))

    print(len(ct_files))
    # Checking for duplicate slices
    series_uids = set()
    z_positions = set()

    for f in ct_files:
        dcm = pydicom.dcmread(f, stop_before_pixels=True)
        series_uids.add(dcm.SeriesInstanceUID)
        z_positions.add(round(float(dcm.ImagePositionPatient[2]), 3))

    print("SeriesInstanceUIDs:", len(series_uids))
    print("Unique Z positions:", len(z_positions))

    # Removing Duplicates

    # deduped_files = {}
    # for f in ct_files:
    #     dcm = pydicom.dcmread(f, stop_before_pixels=True)
    #     z = round(float(dcm.ImagePositionPatient[2]), 3)
    #
    #     # Keep the first slice encountered at each Z
    #     if z not in deduped_files:
    #         deduped_files[z] = f
    #
    # # Sort by Z
    # ct_files = [deduped_files[z] for z in sorted(deduped_files.keys())]

    # print("After deduplication:", len(ct_files))

    # Step 2: Select RTStruct file
    rtstruct_path = select_RTStruct_file()

    if not rtstruct_path:
        print("No RTStruct selected.")
        sys.exit(0)

    data = {
        "dicom_series_path": dicom_series_path,
        "rtstruct_path": rtstruct_path,
        "ct_files": ct_files,
        "additional_target": False
    }
    with open("paths.json", "w") as f:
        json.dump(data, f)

    # Step 3: Pick primary ROI (main target)
    picker = ROIPicker(rtstruct_path)
    picker.show()
    app.exec_()

    if picker.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    primary_roi = picker.selected_roi
    lattice_geometry = picker.lattice_geometry
    Tolerance = picker.tolerance_value

    with open("paths.json", "r") as f:
        data = json.load(f)

    data["lattice_geometry"] = lattice_geometry
    data["Tolerance"] = Tolerance

    with open("paths.json", "w") as f:
        json.dump(data, f)

    print(f"Primary ROI selected: {primary_roi}")

    # Step 4: Optional - Pick secondary ROI
    picker2 = ROIPicker2(rtstruct_path)
    picker2.setWindowTitle("Select Optional Additional Target ROI")
    picker2.show()
    app.exec_()

    if picker2.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    secondary_roi = picker2.selected_roi

    if secondary_roi is not None:
        print(f"Additional ROI selected: {secondary_roi}")
    else:
        print("No additional ROI selected.")

    # Step 5: Select OARs to be given additional margin
    picker3 = OARPicker(rtstruct_path)
    picker3.setWindowTitle("Select OARs and required margin")
    picker3.show()
    app.exec_()

    if picker3.user_cancelled:
        print("No primary ROI selected.")
        sys.exit(0)

    selected_oars = picker3.selected_oars
    print('OARs to avoid:', selected_oars)

    # Step 5: Process CT and RTStruct
    print("Reading CT image...")

    zs = []
    instances = []

    for f in ct_files:
        dcm = pydicom.dcmread(f, stop_before_pixels=True)
        if hasattr(dcm, "ImagePositionPatient"):
            zs.append(dcm.ImagePositionPatient[2])
        if hasattr(dcm, "InstanceNumber"):
            instances.append(dcm.InstanceNumber)

    print("Total DICOM files:", len(ct_files))
    print("Unique Z positions:", len(set(zs)))
    print("Min Z step (mm):", np.min(np.diff(sorted(set(zs)))))

    ct_image = gt.read_dicom(ct_files)

    print("Reading RTStruct...")

    structset = pydicom.dcmread(rtstruct_path)

    # Step 6: Create primary ROI mask
    print("Creating primary ROI mask...")

    # primary_aroi = gt.region_of_interest(structset, primary_roi)
    # # primary_mask = primary_aroi.get_mask(ct_image, corrected=False)
    # # itk.imwrite(primary_mask, "Lattice_target.mhd")  # Save for visualization if needed
    #
    # # Get the raw mask
    # primary_mask_raw = primary_aroi.get_mask(ct_image, corrected=False)
    #
    #
    # # Ensure mask and CT have matching pixel type (float)
    # mask_cast = itk.cast_image_filter(primary_mask_raw, ttype=(type(primary_mask_raw), itk.Image[itk.F, 3]))
    #
    # # Now resample safely
    # primary_mask_resampled = resample_to_reference(mask_cast, ct_image, interpolator='nearest')
    #
    # # Convert back to unsigned char (binary)
    # primary_mask = itk.cast_image_filter(primary_mask_resampled,ttype=(type(primary_mask_resampled), itk.Image[itk.UC, 3]))

    # Define folder path
    mask_folder = Path("Masks")  # relative to your script, or use absolute path

    # Create folder if it doesn't exist
    mask_folder.mkdir(parents=True, exist_ok=True)

    primary_masks = []

    for roi_name in primary_roi:
        primary_aroi = gt.region_of_interest(structset, roi_name)

        # Get raw mask
        primary_mask_raw = primary_aroi.get_mask(ct_image, corrected=False)

        # Cast to float
        mask_cast = itk.cast_image_filter(primary_mask_raw, ttype=(type(primary_mask_raw), itk.Image[itk.F, 3]))

        # Resample to CT
        primary_mask_resampled = resample_to_reference(
            mask_cast, ct_image, interpolator='nearest')

        # Convert back to binary
        primary_mask = itk.cast_image_filter(primary_mask_resampled,
                                             ttype=(type(primary_mask_resampled), itk.Image[itk.UC, 3]))

        # Save each mask individually in a list
        primary_masks.append({"name": roi_name, "mask": primary_mask})

    for entry in primary_masks:
        roi_name = entry["name"]
        mask = entry["mask"]

        # Make a safe filename (replace spaces or special chars)
        safe_name = roi_name.replace(" ", "_").replace("/", "_")
        filename = mask_folder / f"Lattice_target_{safe_name}.mhd"

        itk.imwrite(mask, str(filename))  # str() converts Path to string for ITK
        print(f"Saved mask for ROI {roi_name} -> {filename}")

    # Step 7: If secondary ROI, create secondary mask
    if secondary_roi is not None:
        print("Creating secondary ROI mask...")
        secondary_aroi = gt.region_of_interest(structset, secondary_roi)
        secondary_mask = secondary_aroi.get_mask(ct_image, corrected=False)
        itk.imwrite(secondary_mask, "Additional_target.mhd")  # Save secondary mask separately

        # secondary_mask_raw = secondary_aroi.get_mask(ct_image, corrected=False)
        # secondary_mask = transform.apply(secondary_mask_raw, reference=ct_image, interpolator='nearest')
        # itk.imwrite(secondary_mask, "Additional_target.mhd")

        # Optional: convert to NumPy for later use
        primary_array = itk.GetArrayFromImage(primary_mask)
        secondary_array = itk.GetArrayFromImage(secondary_mask)
    else:
        secondary_mask = None
        secondary_array = None
        print("No secondary ROI selected.")

    # itk.imwrite(primary_mask, "Lattice_target.mhd")

    # Creating combined mask of OARs
    print('Creating combined mask of OARs...')
    oar_mask_index = 0
    if len(selected_oars) != 0:
        oar_masks = []
        spacing = np.array(ct_image.GetSpacing())[::-1]

        for oar_name, margin_mm in selected_oars:
            roi = gt.region_of_interest(structset, oar_name)
            mask_itk = roi.get_mask(ct_image, corrected=False)

            # Convert to NumPy (binary)
            mask_np = itk.GetArrayFromImage(mask_itk).astype(bool)

            # Distance transform OUTSIDE the OAR
            dist = distance_transform_edt(~mask_np, sampling=spacing)

            # Dilate by margin (mm)
            dilated_mask = dist <= margin_mm

            oar_masks.append({
                "name": oar_name,
                "margin_mm": margin_mm,
                "mask": dilated_mask
            })

        oars_combined = None

        for oar in oar_masks:
            arr = oar["mask"].astype(bool)

            if oars_combined is None:
                oars_combined = arr
            else:
                oars_combined = np.logical_or(oars_combined, arr)

        oars_combined_mask = itk.GetImageFromArray(oars_combined.astype(np.uint8))

        oars_combined_mask.SetSpacing(ct_image.GetSpacing())
        oars_combined_mask.SetOrigin(ct_image.GetOrigin())
        oars_combined_mask.SetDirection(ct_image.GetDirection())

        itk.imwrite(oars_combined_mask, "OARs_combined.mhd")

    else:
        print("No OARs selected — using empty OAR mask")
        oars_combined_mask = np.zeros_like(primary_mask_raw, dtype=bool)

    print('OARs combined and saved as OARs_combined.mhd')

    # Step 8: Create distance map
    for entry in primary_masks:
        current_mask = entry["mask"]
        roi = entry["name"]
        primary_array = itk.GetArrayFromImage(current_mask)

        spacing = np.array(ct_image.GetSpacing())

        safe_name = roi.replace(" ", "_").replace("/", "_")
        tumour_filename = mask_folder / f"tumour_volume_{safe_name}.npy"
        distance_map_filename = mask_folder / f"distance_map_{safe_name}.npy"

        # print("CT shape:", itk.GetArrayFromImage(ct_image).shape)
        # print("CT spacing:", ct_image.GetSpacing())
        #
        # print("Primary mask shape:", primary_array.shape)
        # print("Primary mask spacing:", primary_mask.GetSpacing())
        # print("Primary mask origin:", primary_mask.GetOrigin())
        # print("Primary mask direction:\n", primary_mask.GetDirection())

        print("Calculating distance maps...")

        if len(selected_oars) != 0:
            oars_mask = itk.imread("OARs_combined.mhd")
            oars_combined_array = itk.GetArrayFromImage(oars_mask)

            primary_array[:, 0] = (primary_array[:, 0] * spacing[2]).astype(np.uint8)
            primary_array[:, 1] = (primary_array[:, 1] * spacing[1]).astype(np.uint8)
            primary_array[:, 2] = (primary_array[:, 2] * spacing[0]).astype(np.uint8)

            oars_combined_array[:, 0] = (oars_combined_array[:, 0] * spacing[2]).astype(np.uint8)
            oars_combined_array[:, 1] = (oars_combined_array[:, 1] * spacing[1]).astype(np.uint8)
            oars_combined_array[:, 2] = (oars_combined_array[:, 2] * spacing[0]).astype(np.uint8)

            tumour_volume = primary_array & (~oars_combined_array)
            distance_map = distance_transform_edt(tumour_volume)

            OARs_volume = oars_combined_array
            OARs_distance_map = distance_transform_edt(OARs_volume)

            np.save("OARs_volume.npy", OARs_volume)
            np.save("OARs_distance_map.npy", OARs_distance_map)

        else:
            primary_array[:, 0] = (primary_array[:, 0] * spacing[2]).astype(np.uint8)
            primary_array[:, 1] = (primary_array[:, 1] * spacing[1]).astype(np.uint8)
            primary_array[:, 2] = (primary_array[:, 2] * spacing[0]).astype(np.uint8)

            tumour_volume = primary_array
            distance_map = distance_transform_edt(tumour_volume)

        print('reached here')

        np.save(str(tumour_filename), tumour_volume)
        np.save(str(distance_map_filename), distance_map)

    # np.save("lattice_geometry.txt", lattice_geometry)

    print("Distance maps calculated!")

    if secondary_roi is not None:
        print("Saving secondary mask as Additional_target.mhd")
        itk.imwrite(secondary_mask, "Additional_target.mhd")

        # Load and convert secondary mask
        secondary_mask = itk.imread("Additional_target.mhd")
        secondary_array = itk.GetArrayFromImage(secondary_mask)

        secondary_array[:, 0] = (secondary_array[:, 0] * spacing[2]).astype(np.uint8)
        secondary_array[:, 1] = (secondary_array[:, 1] * spacing[1]).astype(np.uint8)
        secondary_array[:, 2] = (secondary_array[:, 2] * spacing[0]).astype(np.uint8)

        additional_target_volume = secondary_array

        np.save("additional_target_volume.npy", additional_target_volume)

        data["additional_target"] = True

        print("Additional target volume saved!")

    else:
        print("No secondary ROI selected.")


if __name__ == "__main__":
    main()


