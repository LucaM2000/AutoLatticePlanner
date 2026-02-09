from pydicom.uid import generate_uid
import glob
import os
import tkinter as tk
from tkinter import filedialog, ttk
import pydicom
from rt_utils import RTStructBuilder
import itk
import numpy as np
import json

from sympy.codegen.ast import continue_


def draw_sphere_mask(mask, center_mm, radius_mm, spacing):
    x_dim, y_dim, z_dim = mask.shape
    z_spacing, y_spacing, x_spacing = spacing

    # Define voxel bounds around the sphere center
    radius_vox = [
        int(np.ceil(radius_mm / x_spacing)),
        int(np.ceil(radius_mm / y_spacing)),
        int(np.ceil(radius_mm / z_spacing))
    ]

    x0 = max(int(round(center_mm[0])) - radius_vox[0], 0)
    x1 = min(int(round(center_mm[0])) + radius_vox[0] + 1, x_dim)
    y0 = max(int(round(center_mm[1])) - radius_vox[1], 0)
    y1 = min(int(round(center_mm[1])) + radius_vox[1] + 1, y_dim)
    z0 = max(int(round(center_mm[2])) - radius_vox[2], 0)
    z1 = min(int(round(center_mm[2])) + radius_vox[2] + 1, z_dim)

    X, Y, Z = np.meshgrid(
        np.arange(x0, x1),
        np.arange(y0, y1),
        np.arange(z0, z1),
        indexing='ij'
    )

    dist_vox = np.sqrt(
        ((X - center_mm[0]) * x_spacing) ** 2 +
        ((Y - center_mm[1]) * y_spacing) ** 2 +
        ((Z - center_mm[2]) * z_spacing) ** 2
    )

    mask[x0:x1, y0:y1, z0:z1][dist_vox <= radius_mm] = 1


def main():

    print('Loading data...')





    with open("paths.json", "r") as f:
        data = json.load(f)

    lattice_geometry = data["lattice_geometry"]


    if lattice_geometry == 'Normal (Duriseti)':
        sph_radius = 7.5

    else:
        sph_radius = 10


    def get_rtstruct_labels():
        result = {}

        def submit():
            result["rtstruct_series_desc"] = rtstruct_series_desc.get()
            result["ds_series_desc"] = ds_series_desc.get()
            result["structure_set_label"] = structure_set_label.get()
            result["structure_set_name"] = structure_set_name.get()
            result["series_number"] = int(series_number.get())
            root.destroy()

        root = tk.Tk()
        root.title("RTSTRUCT Label Editor")
        root.resizable(False, False)

        padding = {"padx": 10, "pady": 5}

        # Variables with defaults
        rtstruct_series_desc = tk.StringVar(value="Lattice Generator")
        ds_series_desc = tk.StringVar(value="LucaScript")
        structure_set_label = tk.StringVar(value="Luca-Lat")
        structure_set_name = tk.StringVar(value="Luca-Lat")
        series_number = tk.StringVar(value="990")

        # Layout
        ttk.Label(root, text="RTSTRUCT Series Description").grid(row=0, column=0, sticky="w", **padding)
        ttk.Entry(root, textvariable=rtstruct_series_desc, width=30).grid(row=0, column=1, **padding)

        ttk.Label(root, text="DS Series Description").grid(row=1, column=0, sticky="w", **padding)
        ttk.Entry(root, textvariable=ds_series_desc, width=30).grid(row=1, column=1, **padding)

        ttk.Label(root, text="Structure Set Label").grid(row=2, column=0, sticky="w", **padding)
        ttk.Entry(root, textvariable=structure_set_label, width=30).grid(row=2, column=1, **padding)

        ttk.Label(root, text="Structure Set Name").grid(row=3, column=0, sticky="w", **padding)
        ttk.Entry(root, textvariable=structure_set_name, width=30).grid(row=3, column=1, **padding)

        ttk.Label(root, text="Series Number").grid(row=4, column=0, sticky="w", **padding)
        ttk.Entry(root, textvariable=series_number, width=30).grid(row=4, column=1, **padding)

        ttk.Button(root, text="Apply", command=submit).grid(row=5, column=0, columnspan=2, pady=15)

        root.mainloop()
        return result


    spheres_inside = np.loadtxt("spheres_inside.txt")

    with open("paths.json", "r") as f:
        data = json.load(f)

    dicom_series_path = data["dicom_series_path"]
    rtstruct_path = data["rtstruct_path"]
    ct_files = data["ct_files"]

    mask_1 = itk.imread("all_spheres_original_1.mhd")
    mask_2 = itk.imread("all_spheres_expanded_1.mhd")
    mask_array_1 = itk.GetArrayFromImage(mask_1)  # Shape: [z, y, x]
    mask_array_2 = itk.GetArrayFromImage(mask_2)
    spacing = mask_1.GetSpacing()
    origin = mask_1.GetOrigin()
    direction = mask_1.GetDirection()

    colors = [
        [255,   0,   0],   # red
        [0,   255,   0],   # green
        [0,     0, 255],   # blue
        [255, 255,   0],   # yellow
        [255,   0, 255],   # magenta
        [0,   255, 255],   # cyan
        [255, 128,   0],   # orange
        [128,   0, 255],   # purple
        [0,   128, 255],   # sky blue
        [128, 255,   0],   # lime
        [255,   0, 128],   # pink
        [0,   255, 128],   # mint
        [255,  64,  64],   # soft red
        [64,  255,  64],   # soft green
        [64,   64, 255],   # soft blue
        [255, 192,   0],   # gold
        [0,   192, 192],   # teal
        [192,   0, 192],   # violet
        [160,  82,  45],   # brown (useful for bone-like structures)
        [220, 220, 220],   # light grey (PTV shells, margins)
        [128, 128, 128],   # mid grey
    ]

    print('Data Loaded!')

    print('Copying ROI Structures...')

    original_rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path,
        rt_struct_path=rtstruct_path
    )



    # Create new RTStruct to modify

    print("CT files passed to RTStruct:", len(ct_files))
    print("Directory CT files:", len(glob.glob(os.path.join(dicom_series_path, "*.dcm"))))

    rtstruct_new = RTStructBuilder.create_new(dicom_series_path)

    # Copy all ROIs from original RTStruct
    for roi_name in original_rtstruct.get_roi_names():
        mask = original_rtstruct.get_roi_mask_by_name(roi_name)
        rtstruct_new.add_roi(mask=mask, name=roi_name)
    # Add binary mask

    print('Structures copied!')

    print('Adding sphere grid...')

    mask_bool_1 = mask_array_1.astype(bool)
    mask_bool_2 = mask_array_2.astype(bool)

    mask_bool_1 = np.transpose(mask_bool_1, (1, 2, 0))
    mask_bool_2 = np.transpose(mask_bool_2, (1, 2, 0))

    print("Mask shape:", mask_bool_1.shape)
    print("CT slices:", len(ct_files))

    # Add your new ROIs
    rtstruct_new.add_roi(mask=mask_bool_1, name="Cibles_Lattice", color=[255, 0, 0])
    rtstruct_new.add_roi(mask=mask_bool_2, name="CiblesExt_Lattice", color=[0, 255, 0])

    print('Grid added!')

    # rtstruct.add_roi(mask=mask_bool_1, name="Cibles", color=[255, 0, 0])
    # rtstruct.add_roi(mask=mask_bool_2, name="CiblesExtended", color=[0 ,255, 0])

    # individual_sphere_structrues = input('Would you like to save individual spheres? (y/n):')
    individual_sphere_structrues = 'y'

    if individual_sphere_structrues == 'y':
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
            draw_sphere_mask(mask2, center_mm, sph_radius +5, spacing)

            # Convert to bool and transpose to match RTStruct convention
            single_mask_bool2 = np.transpose(mask2.astype(bool), (1, 2, 0))

            # Pick color from palette (wrap around if more spheres than colors)
            color = colors[(i - 1) % len(colors)]

            # Add ROI with unique name
            rtstruct_new.add_roi(mask=single_mask_bool, name=f"Sphere_Ext {i}", color=color)

        print('Individual sphere structures saved!')

    else:
        print('Not saving individual spheres')

    # print("DEBUG: reached file save section")

    # # Ask user for save location and filename
    # # --- Fix for Tkinter dialog freezing ---
    # root = tk.Tk()
    # root.withdraw()
    # root.update()  # Force Tk to process pending events before opening dialog
    #
    # file_path = filedialog.asksaveasfilename(
    #     defaultextension=".dcm",
    #     filetypes=[("DICOM files", "*.dcm")],
    #     title="Save RTSTRUCT As"
    # )
    #
    # root.destroy()  # Clean up properly after dialog closes
    #
    # if not file_path:
    #     print("Save cancelled.")
    #     exit()


    print("DEBUG: reached file save section")

    # 1️⃣ Ask for file path
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".dcm",
            filetypes=[("DICOM files", "*.dcm")],
            title="Save RTSTRUCT As"
        )
        root.destroy()
    except Exception as e:
        print("⚠️ File dialog failed, using default path instead:", e)
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop, "default_rtstruct.dcm")

    if not file_path:
        print("Save cancelled or no file path returned.")
        exit()

    print("DEBUG: file path selected ->", file_path)

    # Save first
    rtstruct_new.save(file_path)

    # Now read back
    ds = pydicom.dcmread(file_path)

    labels = get_rtstruct_labels()

    ds.SeriesDescription = labels["ds_series_desc"]
    ds.StructureSetLabel = labels["structure_set_label"]
    ds.StructureSetName  = labels["structure_set_name"]
    ds.SeriesNumber      = labels["series_number"]

    from pydicom.uid import generate_uid
    ds.SOPInstanceUID    = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.save_as(file_path)

if __name__ == "__main__":
    main()