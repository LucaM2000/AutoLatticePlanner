import glob
import json
import os
from pathlib import Path

import gatetools as gt
import itk
import numpy as np
import pydicom
from scipy.ndimage import distance_transform_edt


class LatticeData:
    """Class for storing lattice-related data with JSON serialization."""

    def __init__(
        self,
        dicom_series_path=None,
        rtstruct_path=None,
        additional_target=False,
        lattice_geometry=None,
        tolerance=None,
        primary_roi=None,
        secondary_roi=None,
        selected_oars=None,
        output_folder=".",
    ):
        self.dicom_series_path = dicom_series_path
        self.rtstruct_path = rtstruct_path
        self.additional_target = additional_target
        self.lattice_geometry = lattice_geometry
        self.tolerance = tolerance
        self.primary_roi = primary_roi
        self.secondary_roi = secondary_roi
        self.selected_oars = selected_oars
        self.output_folder = output_folder

    def to_dict(self):
        return {
            "dicom_series_path": self.dicom_series_path,
            "rtstruct_path": self.rtstruct_path,
            "additional_target": self.additional_target,
            "lattice_geometry": self.lattice_geometry,
            "tolerance": self.tolerance,
            "primary_roi": self.primary_roi,
            "secondary_roi": self.secondary_roi,
            "selected_oars": self.selected_oars,
            "output_folder": self.output_folder,
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data):
        return cls(
            dicom_series_path=data.get("dicom_series_path"),
            rtstruct_path=data.get("rtstruct_path"),
            additional_target=data.get("additional_target"),
            lattice_geometry=data.get("lattice_geometry"),
            tolerance=data.get("tolerance"),
            primary_roi=data.get("primary_roi"),
            secondary_roi=data.get("secondary_roi"),
            selected_oars=data.get("selected_oars"),
            output_folder=data.get("output_folder"),
        )

    @classmethod
    def from_json(cls, json_string):
        return cls.from_dict(json.loads(json_string))


def resample_to_reference(input_image, reference_image, interpolator="nearest"):
    """
    Resample input_image to match reference_image geometry (spacing, size, origin, direction)
    """

    # Set up the resampler
    ResampleFilterType = itk.ResampleImageFilter[
        type(input_image), type(reference_image)
    ]
    resampler = ResampleFilterType.New()

    resampler.SetInput(input_image)
    resampler.SetReferenceImage(reference_image)
    resampler.UseReferenceImageOn()

    # Choose interpolator
    if interpolator == "nearest":
        InterpolatorType = itk.NearestNeighborInterpolateImageFunction[
            type(reference_image), itk.D
        ]
        interpolator_obj = InterpolatorType.New()
    else:
        InterpolatorType = itk.LinearInterpolateImageFunction[
            type(reference_image), itk.D
        ]
        interpolator_obj = InterpolatorType.New()

    resampler.SetInterpolator(interpolator_obj)

    # Execute the resampling
    resampler.Update()
    return resampler.GetOutput()


def compute_volumes(data):
    """
    data: LatticeData
    """

    dicom_series_path = data.dicom_series_path
    rtstruct_path = data.rtstruct_path
    primary_roi = data.primary_roi
    secondary_roi = data.secondary_roi
    selected_oars = data.selected_oars
    output_folder = Path(data.output_folder)

    # Process CT and RTStruct
    print("Reading CT image...")

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

    primary_aroi = gt.region_of_interest(structset, primary_roi)
    # primary_mask = primary_aroi.get_mask(ct_image, corrected=False)
    # itk.imwrite(primary_mask, "Lattice_target.mhd")  # Save for visualization if needed

    # Get the raw mask
    primary_mask_raw = primary_aroi.get_mask(ct_image, corrected=False)

    # primary_mask_raw = itk.imread("C:\\Users\\MULCAHY\\Documents\\1 RTSTRUCT SS_061025iiMAR-GVTLattice-label.mhd")

    # Ensure mask and CT have matching pixel type (float)
    mask_cast = itk.cast_image_filter(
        primary_mask_raw, ttype=(type(primary_mask_raw), itk.Image[itk.F, 3])
    )

    # Now resample safely
    primary_mask_resampled = resample_to_reference(
        mask_cast, ct_image, interpolator="nearest"
    )

    # Convert back to unsigned char (binary)
    primary_mask = itk.cast_image_filter(
        primary_mask_resampled,
        ttype=(type(primary_mask_resampled), itk.Image[itk.UC, 3]),
    )

    # Save for visualization
    print("Saving primary target mask...")
    itk.imwrite(primary_mask, output_folder / "Lattice_target.mhd")
    itk.imwrite(primary_mask_raw, output_folder / "Lattice_raw.mhd")
    print("Primary mask saved as Lattice_target.mhd")

    # Step 7: If secondary ROI, create secondary mask
    primary_array = itk.GetArrayFromImage(primary_mask)
    if secondary_roi is not None:
        print("Creating secondary ROI mask...")
        secondary_aroi = gt.region_of_interest(structset, secondary_roi)
        secondary_mask = secondary_aroi.get_mask(ct_image, corrected=False)
        itk.imwrite(
            secondary_mask, output_folder / "Additional_target.mhd"
        )  # Save secondary mask separately

        # secondary_mask_raw = secondary_aroi.get_mask(ct_image, corrected=False)
        # secondary_mask = transform.apply(secondary_mask_raw, reference=ct_image, interpolator='nearest')
        # itk.imwrite(secondary_mask, "Additional_target.mhd")

        # Optional: convert to NumPy for later use
        secondary_array = itk.GetArrayFromImage(secondary_mask)
    else:
        secondary_mask = None
        secondary_array = None
        print("No secondary ROI selected.")

    # itk.imwrite(primary_mask, "Lattice_target.mhd")

    # Creating combined mask of OARs
    print("Creating combined mask of OARs...")
    if len(selected_oars) != 0:
        oar_masks = []
        spacing = np.array(ct_image.GetSpacing())

        for oar_name, margin_mm in selected_oars:
            roi = gt.region_of_interest(structset, oar_name)
            mask_itk = roi.get_mask(ct_image, corrected=False)

            # Convert to NumPy (binary)
            mask_np = itk.GetArrayFromImage(mask_itk).astype(bool)

            # Distance transform OUTSIDE the OAR
            dist = distance_transform_edt(~mask_np, sampling=spacing)

            # Dilate by margin (mm)
            dilated_mask = dist <= margin_mm

            oar_masks.append(
                {"name": oar_name, "margin_mm": margin_mm, "mask": dilated_mask}
            )

        oars_combined = None

        for oar in oar_masks:
            arr = oar["mask"].astype(bool)

            if oars_combined is None:
                oars_combined = arr
            else:
                oars_combined = np.logical_or(oars_combined, arr)

        oars_combined_mask = itk.GetImageFromArray(oars_combined.astype(np.uint8))

        oars_combined_mask.CopyInformation(ct_image)

        itk.imwrite(oars_combined_mask, output_folder / "OARs_combined.mhd")

    else:
        print("No OARs selected — using empty OAR mask")
        oars_combined_mask = np.zeros_like(primary_mask_raw, dtype=bool)

    print("OARs combined and saved as OARs_combined.mhd")

    # Step 8: Create distance map
    spacing = np.array(ct_image.GetSpacing())

    print("CT shape:", itk.GetArrayFromImage(ct_image).shape)
    print("CT spacing:", ct_image.GetSpacing())

    print("Primary mask shape:", primary_array.shape)
    print("Primary mask spacing:", primary_mask.GetSpacing())
    print("Primary mask origin:", primary_mask.GetOrigin())
    print("Primary mask direction:\n", primary_mask.GetDirection())

    print("Calculating distance maps...")

    if len(selected_oars) != 0:

        oars_combined_array = itk.GetArrayFromImage(oars_combined_mask)

        primary_array[:, 0] = (primary_array[:, 0] * spacing[2]).astype(np.uint8)
        primary_array[:, 1] = (primary_array[:, 1] * spacing[1]).astype(np.uint8)
        primary_array[:, 2] = (primary_array[:, 2] * spacing[0]).astype(np.uint8)

        oars_combined_array[:, 0] = (oars_combined_array[:, 0] * spacing[2]).astype(
            np.uint8
        )
        oars_combined_array[:, 1] = (oars_combined_array[:, 1] * spacing[1]).astype(
            np.uint8
        )
        oars_combined_array[:, 2] = (oars_combined_array[:, 2] * spacing[0]).astype(
            np.uint8
        )

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

    print("reached here")

    np.save(output_folder / "tumour_volume.npy", tumour_volume)
    np.save(output_folder / "distance_map.npy", distance_map)

    # np.save("lattice_geometry.txt", lattice_geometry)

    print("Distance maps calculated!")

    if secondary_roi is not None:
        print("Saving secondary mask as Additional_target.mhd")
        itk.imwrite(secondary_mask, output_folder / "Additional_target.mhd")

        # Load and convert secondary mask
        secondary_array = itk.GetArrayFromImage(secondary_mask)

        secondary_array[:, 0] = (secondary_array[:, 0] * spacing[2]).astype(np.uint8)
        secondary_array[:, 1] = (secondary_array[:, 1] * spacing[1]).astype(np.uint8)
        secondary_array[:, 2] = (secondary_array[:, 2] * spacing[0]).astype(np.uint8)

        additional_target_volume = secondary_array

        np.save(
            output_folder / "additional_target_volume.npy", additional_target_volume
        )

        print("Additional target volume saved!")

    else:
        print("No secondary ROI selected.")
