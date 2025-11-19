from pydicom.uid import generate_uid
import tkinter as tk
from tkinter import filedialog
import pydicom
from rt_utils import RTStructBuilder
import itk
import numpy as np
import json

from sympy.codegen.ast import continue_

print('Loading data...')

from Lattice_settings_and_functions import draw_sphere_mask
from Lattice_settings_and_functions import sph_radius

spheres_inside = np.loadtxt("spheres_inside.txt")

with open("paths.json", "r") as f:
    data = json.load(f)

dicom_series_path = data["dicom_series_path"]
rtstruct_path = data["rtstruct_path"]
ct_files = data["ct_files"]

mask_1 = itk.imread("all_spheres_mask_original_1.mhd")
mask_2 = itk.imread("all_spheres_mask_expanded_1.mhd")
mask_array_1 = itk.GetArrayFromImage(mask_1)  # Shape: [z, y, x]
mask_array_2 = itk.GetArrayFromImage(mask_2)
spacing = mask_1.GetSpacing()
origin = mask_1.GetOrigin()
direction = mask_1.GetDirection()

colors = [
    [255, 0, 0],    # red
    [0, 255, 0],    # green
    [0, 0, 255],    # blue
    [255, 255, 0],  # yellow
    [255, 0, 255],  # magenta
    [0, 255, 255],  # cyan
]

print('Data Loaded!')

print('Copying ROI Structures...')

original_rtstruct = RTStructBuilder.create_from(
    dicom_series_path=dicom_series_path,
    rt_struct_path=rtstruct_path
)



# Create new RTStruct to modify
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


# Add your new ROIs
rtstruct_new.add_roi(mask=mask_bool_1, name="Cibles", color=[255, 0, 0])
rtstruct_new.add_roi(mask=mask_bool_2, name="CiblesExt", color=[0, 255, 0])

print('Grid added!')

# rtstruct.add_roi(mask=mask_bool_1, name="Cibles", color=[255, 0, 0])
# rtstruct.add_roi(mask=mask_bool_2, name="CiblesExtended", color=[0 ,255, 0])

individual_sphere_structrues = input('Would you like to save individual spheres? (y/n):')

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

# Try to open file dialog safely
try:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()  # Helps avoid hanging in some environments

    file_path = filedialog.asksaveasfilename(
        defaultextension=".dcm",
        filetypes=[("DICOM files", "*.dcm")],
        title="Save RTSTRUCT As"
    )
    root.destroy()
except Exception as e:
    print("⚠️ File dialog failed, using default path instead:", e)
    file_path = r"C:\Users\MULCAHY\Desktop\default_rtstruct.dcm"

# Fallback or cancel behaviour
if not file_path:
    print("Save cancelled or no file path returned.")
    exit()

print("DEBUG: file path selected ->", file_path)

rtstruct_new.save(file_path)




ds = pydicom.dcmread(file_path)

# Change the Series Description
rtstruct_new.SeriesDescription = "LucaScript"
ds.SeriesDescription = "LucaScript"
ds.StructureSetLabel = "Luca-Lat"
ds.StructureSetName  = "Luca-Lat"
ds.SeriesNumber      = 990

# Generate and assign new UIDs
ds.SOPInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()

#
#
ds.save_as(file_path)

print("Saved with new UIDs!")