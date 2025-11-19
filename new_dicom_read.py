import gatetools as gt
import numpy as np
import glob
import pydicom
import itk
from scipy.ndimage import distance_transform_edt
import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QListWidget,QPushButton, QLabel, QFileDialog, QMessageBox, QHBoxLayout)
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


class ROIPicker(QWidget):
    def __init__(self, rtstruct_path):
        super().__init__()
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


def main():
    app = QApplication(sys.argv)

    # Step 1: Select CT folder
    dicom_series_path = select_CT_folder()
    if not dicom_series_path:
        print("No CT folder selected.")
        return
    dicom_series_path = os.path.normpath(dicom_series_path)
    ct_files = glob.glob(os.path.join(dicom_series_path, "*.dcm"))

    # Step 2: Select RTStruct file
    rtstruct_path = select_RTStruct_file()
    if not rtstruct_path:
        print("No RTStruct selected.")
        return

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

    if not picker.selected_roi:
        print("No primary ROI selected.")
        return

    primary_roi = picker.selected_roi
    print(f"Primary ROI selected: {primary_roi}")

    # Step 4: Optional - Pick secondary ROI
    picker2 = ROIPicker(rtstruct_path)
    picker2.setWindowTitle("Select Optional Additional Target ROI")
    picker2.show()
    app.exec_()

    secondary_roi = picker2.selected_roi
    if secondary_roi is not None:
        print(f"Additional ROI selected: {secondary_roi}")
    else:
        print("No additional ROI selected.")

    # Step 5: Process CT and RTStruct
    print("Reading CT image...")
    ct_image = gt.read_dicom(ct_files)

    print("Reading RTStruct...")

    structset = pydicom.dcmread(rtstruct_path)

    # Step 6: Create primary ROI mask
    print("Creating primary ROI mask...")



    primary_aroi = gt.region_of_interest(structset, primary_roi)
    # primary_mask = primary_aroi.get_mask(ct_image, corrected=False)
    # itk.imwrite(primary_mask, "Lattice_target.mhd")  # Save for visualization if needed

    # Get the raw mask
    primary_mask_raw = primary_aroi.get_mask(ct_image, corrected=False)

    # primary_mask_raw = itk.imread("C:\\Users\\MULCAHY\\Documents\\1 RTSTRUCT SS_061025iiMAR-GVTLattice-label.mhd")

    # Ensure mask and CT have matching pixel type (float)
    mask_cast = itk.cast_image_filter(primary_mask_raw, ttype=(type(primary_mask_raw), itk.Image[itk.F, 3]))

    # Now resample safely
    primary_mask_resampled = resample_to_reference(mask_cast, ct_image, interpolator='nearest')

    # Convert back to unsigned char (binary)
    primary_mask = itk.cast_image_filter(primary_mask_resampled,ttype=(type(primary_mask_resampled), itk.Image[itk.UC, 3]))


    # Save for visualization
    itk.imwrite(primary_mask, "Lattice_target.mhd")
    itk.imwrite(primary_mask_raw, "Lattice_raw.mhd")
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

    print("Saving mask as Lattice_target.mhd")
    # itk.imwrite(primary_mask, "Lattice_target.mhd")

    # Step 8: Create distance map
    primary_mask = itk.imread("Lattice_target.mhd")
    primary_array = itk.GetArrayFromImage(primary_mask)

    secondary_mask = itk.imread('Additional_target.mhd')
    secondary_array = itk.GetArrayFromImage(secondary_mask)

    spacing = np.array(ct_image.GetSpacing())

    print("Calculating distance map...")

    primary_array[:, 0] = (primary_array[:, 0] * spacing[2]).astype(np.uint8)
    primary_array[:, 1] = (primary_array[:, 1] * spacing[1]).astype(np.uint8)
    primary_array[:, 2] = (primary_array[:, 2] * spacing[0]).astype(np.uint8)

    tumour_volume = primary_array
    distance_map = distance_transform_edt(tumour_volume)
    np.save("tumour_volume.npy", tumour_volume)
    np.save("distance_map.npy", distance_map)

    print("Distance map calculated!")

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