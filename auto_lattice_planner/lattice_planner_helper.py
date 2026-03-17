import glob
import json
import os
from pathlib import Path

import gatetools as gt
import itk
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from rt_utils import RTStructBuilder
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
    print("Creating primary ROI masks...")

    # Define folder path
    mask_folder = Path("Masks")  

    # Create folder if it doesn't exist
    mask_folder.mkdir(parents=True, exist_ok=True)

    # Empty folder
    for filename in os.listdir(mask_folder):
        file_path = os.path.join(mask_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete subfolder
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

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
    for entry in primary_masks:
        current_mask = entry["mask"]
        roi = entry["name"]
        primary_array = itk.GetArrayFromImage(current_mask)

        spacing = np.array(ct_image.GetSpacing())

        safe_name = roi.replace(" ", "_").replace("/", "_")
        tumour_filename = mask_folder / f"tumour_volume_{safe_name}.npy"
        distance_map_filename = mask_folder / f"distance_map_{safe_name}.npy"

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
    
        print("reached here")
    
        np.save(output_folder / str(tumour_filename), tumour_volume)
        np.save(output_folder / str(distance_map_filename), distance_map)
    
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


def draw_sphere_mask(mask, center_mm, radius_mm, spacing):
    x_dim, y_dim, z_dim = mask.shape
    z_spacing, y_spacing, x_spacing = spacing

    # Define voxel bounds around the sphere center
    radius_vox = [
        int(np.ceil(radius_mm / x_spacing)),
        int(np.ceil(radius_mm / y_spacing)),
        int(np.ceil(radius_mm / z_spacing)),
    ]

    x0 = max(int(round(center_mm[0])) - radius_vox[0], 0)
    x1 = min(int(round(center_mm[0])) + radius_vox[0] + 1, x_dim)
    y0 = max(int(round(center_mm[1])) - radius_vox[1], 0)
    y1 = min(int(round(center_mm[1])) + radius_vox[1] + 1, y_dim)
    z0 = max(int(round(center_mm[2])) - radius_vox[2], 0)
    z1 = min(int(round(center_mm[2])) + radius_vox[2] + 1, z_dim)

    X, Y, Z = np.meshgrid(
        np.arange(x0, x1), np.arange(y0, y1), np.arange(z0, z1), indexing="ij"
    )

    dist_vox = np.sqrt(
        ((X - center_mm[0]) * x_spacing) ** 2
        + ((Y - center_mm[1]) * y_spacing) ** 2
        + ((Z - center_mm[2]) * z_spacing) ** 2
    )

    mask[x0:x1, y0:y1, z0:z1][dist_vox <= radius_mm] = 1


def convert_to_rtstruct(data):
    print("Loading data...")

    lattice_geometry = data.lattice_geometry
    output = Path(data.output_folder)

    if lattice_geometry == "Normal (Duriseti)":
        sph_radius = 7.5

    else:
        sph_radius = 10

    spheres_inside = np.loadtxt(output / "spheres_inside.txt")

    dicom_series_path = data.dicom_series_path
    rtstruct_path = data.rtstruct_path
    ct_files = glob.glob(os.path.join(dicom_series_path, "*.dcm"))

    mask_1 = itk.imread(output / "all_spheres_original_1.mhd")
    mask_2 = itk.imread(output / "all_spheres_expanded_1.mhd")
    mask_array_1 = itk.GetArrayFromImage(mask_1)  # Shape: [z, y, x]
    mask_array_2 = itk.GetArrayFromImage(mask_2)
    spacing = mask_1.GetSpacing()

    colors = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [255, 255, 0],  # yellow
        [255, 0, 255],  # magenta
        [0, 255, 255],  # cyan
        [255, 128, 0],  # orange
        [128, 0, 255],  # purple
        [0, 128, 255],  # sky blue
        [128, 255, 0],  # lime
        [255, 0, 128],  # pink
        [0, 255, 128],  # mint
        [255, 64, 64],  # soft red
        [64, 255, 64],  # soft green
        [64, 64, 255],  # soft blue
        [255, 192, 0],  # gold
        [0, 192, 192],  # teal
        [192, 0, 192],  # violet
        [160, 82, 45],  # brown (useful for bone-like structures)
        [220, 220, 220],  # light grey (PTV shells, margins)
        [128, 128, 128],  # mid grey
    ]

    print("Data Loaded!")

    print("Copying ROI Structures...")

    original_rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path, rt_struct_path=rtstruct_path
    )

    # Create new RTStruct to modify

    print("CT files passed to RTStruct:", len(ct_files))
    print(
        "Directory CT files:", len(glob.glob(os.path.join(dicom_series_path, "*.dcm")))
    )

    rtstruct_new = RTStructBuilder.create_new(dicom_series_path)

    # Copy all ROIs from original RTStruct
    for roi_name in original_rtstruct.get_roi_names():
        mask = original_rtstruct.get_roi_mask_by_name(roi_name)
        rtstruct_new.add_roi(mask=mask, name=roi_name)
    # Add binary mask

    print("Structures copied!")

    print("Adding sphere grid...")

    mask_bool_1 = mask_array_1.astype(bool)
    mask_bool_2 = mask_array_2.astype(bool)

    mask_bool_1 = np.transpose(mask_bool_1, (1, 2, 0))
    mask_bool_2 = np.transpose(mask_bool_2, (1, 2, 0))

    print("Mask shape:", mask_bool_1.shape)
    print("CT slices:", len(ct_files))

    # Add your new ROIs
    rtstruct_new.add_roi(mask=mask_bool_1, name="Cibles_Lattice", color=[255, 0, 0])
    rtstruct_new.add_roi(mask=mask_bool_2, name="CiblesExt_Lattice", color=[0, 255, 0])

    print("Grid added!")

    # rtstruct.add_roi(mask=mask_bool_1, name="Cibles", color=[255, 0, 0])
    # rtstruct.add_roi(mask=mask_bool_2, name="CiblesExtended", color=[0 ,255, 0])

    # individual_sphere_structrues = input('Would you like to save individual spheres? (y/n):')
    individual_sphere_structrues = "y"

    if individual_sphere_structrues == "y":
        for i, (x, y, z) in enumerate(spheres_inside, start=1):
            center_mm = np.array([float(x), float(y), float(z)])

            # Build one mask just for this sphere
            mask = np.zeros_like(mask_array_1)
            draw_sphere_mask(mask, center_mm, sph_radius, spacing)

            # Convert to bool and transpose to match RTStruct convention
            single_mask_bool = np.transpose(mask.astype(bool), (1, 2, 0))

            # Pick color from palette (wrap around if more spheres than colors)
            color = colors[(i - 1) % len(colors)]

            # Add ROI with unique name
            rtstruct_new.add_roi(mask=single_mask_bool, name=f"Sphere {i}", color=color)

            # Build one mask just for this sphere
            mask2 = np.zeros_like(mask_array_1)
            draw_sphere_mask(mask2, center_mm, sph_radius + 5, spacing)

            # Pick color from palette (wrap around if more spheres than colors)
            color = colors[(i - 1) % len(colors)]

            # Add ROI with unique name
            rtstruct_new.add_roi(
                mask=single_mask_bool, name=f"Sphere_Ext {i}", color=color
            )

        print("Individual sphere structures saved!")

    else:
        print("Not saving individual spheres")
    return rtstruct_new


def save_rtstruct(rtstruct_new, file_path, labels):
    # Save first
    rtstruct_new.save(file_path)

    # Now read back
    ds = pydicom.dcmread(file_path)

    ds.SeriesDescription = labels["ds_series_desc"]
    ds.StructureSetLabel = labels["structure_set_label"]
    ds.StructureSetName = labels["structure_set_name"]
    ds.SeriesNumber = labels["series_number"]

    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.save_as(file_path)
