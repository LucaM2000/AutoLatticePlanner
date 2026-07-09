import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from scipy.ndimage import map_coordinates
import cv2
import gatetools as gt
import itk
import pydicom
import PyQt5.QtCore
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


def select_CT_folder():
    folder = QFileDialog.getExistingDirectory(None, "Select CT Folder")
    return folder

def select_RTStruct_file():
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select RTStruct DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
    )
    return file_path

def select_RTDose_file():
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select RTDose DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
    )
    return file_path


def get_sphere_centres(rtstruct_path, sphere_roi_name, reference_folder):
    """
    Extract sphere centres by clustering contours in 3D using centroid proximity.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(reference_folder)
    reader.SetFileNames(dicom_names)
    reference_image = reader.Execute()

    rtstruct = pydicom.dcmread(rtstruct_path)

    roi_number = next(
        roi.ROINumber for roi in rtstruct.StructureSetROISequence
        if roi.ROIName == sphere_roi_name
    )
    roi_contour = next(
        rc for rc in rtstruct.ROIContourSequence
        if rc.ReferencedROINumber == roi_number
    )

    # Collect all contour centroids in 3D physical space
    contour_centroids = []
    contour_sizes     = []

    for contour in roi_contour.ContourSequence:
        points = np.array(contour.ContourData).reshape(-1, 3)
        cx = float(np.mean(points[:, 0]))
        cy = float(np.mean(points[:, 1]))
        cz = float(points[0, 2])
        contour_centroids.append((cx, cy, cz))
        contour_sizes.append(len(points))

    contour_centroids = np.array(contour_centroids)  # (N, 3)
    contour_sizes     = np.array(contour_sizes)

    # Print a summary to help diagnose clustering
    # print(f"[DEBUG] Total contour slices in ROI: {len(contour_centroids)}")
    # print(f"[DEBUG] Z range: {contour_centroids[:,2].min():.1f} → {contour_centroids[:,2].max():.1f} mm")
    # unique_z = np.unique(contour_centroids[:, 2])
    # print(f"[DEBUG] Unique z values ({len(unique_z)}): {unique_z}")

    # ── Cluster in 3D using a simple greedy approach ──────────────────
    # Two contour centroids belong to the same sphere if their XY distance
    # is less than cluster_xy_mm — this separates spheres that overlap in Z
    cluster_xy_mm = 20.0   # mm — adjust if spheres are very close together in XY

    assigned  = [-1] * len(contour_centroids)
    cluster_id = 0

    for i in range(len(contour_centroids)):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, len(contour_centroids)):
            if assigned[j] != -1:
                continue
            xy_dist = np.linalg.norm(contour_centroids[i, :2] - contour_centroids[j, :2])
            if xy_dist < cluster_xy_mm:
                assigned[j] = cluster_id
        cluster_id += 1

    assigned = np.array(assigned)
    print(f"[DEBUG] Clusters found: {cluster_id}")

    # ── For each cluster, pick the contour at the equator (largest) ──
    spheres = []
    for cid in range(cluster_id):
        idx      = np.where(assigned == cid)[0]
        best_idx = idx[np.argmax(contour_sizes[idx])]
        cx, cy, cz = contour_centroids[best_idx]
        spheres.append((cx, cy, cz))
        # print(f"[DEBUG] Sphere {cid}: centre=({cx:.1f}, {cy:.1f}, {cz:.1f}) mm, "
        #       f"n_slices={len(idx)}, equator_points={contour_sizes[best_idx]}")

    print(f"[INFO] Found {len(spheres)} sphere centres")
    return spheres


def get_dose_profile_between_points(dose_array, dose_sitk, p1_mm, p2_mm, n_samples=100):
    """
    Sample dose along a line between two physical points (mm).
    Returns array of dose values of length n_samples.
    """
    spacing = np.array(dose_sitk.GetSpacing())   # (x, y, z)
    origin  = np.array(dose_sitk.GetOrigin())    # (x, y, z)

    # Convert mm → voxel indices (continuous)
    # dose_array is (z, y, x) so we need to reverse
    def mm_to_voxel(pt_mm):
        pt = (np.array(pt_mm) - origin) / spacing  # (x, y, z) voxel coords
        return np.array([pt[2], pt[1], pt[0]])      # → (z, y, x) for array indexing

    v1 = mm_to_voxel(p1_mm)
    v2 = mm_to_voxel(p2_mm)

    # Interpolation coordinates along the line
    coords = np.array([
        np.linspace(v1[i], v2[i], n_samples) for i in range(3)
    ])  # shape (3, n_samples)

    profile = map_coordinates(dose_array, coords, order=1, mode='nearest')
    return profile


def compute_sphere_profiles(
    dose_array, dose_sitk, sphere_centres,
    neighbour_radius_mm=45, n_samples=100
):
    """
    For each unique pair of spheres within neighbour_radius_mm,
    extract the dose profile and normalise x-axis to [0, 1].
    Returns list of profiles (each length n_samples).
    """
    profiles = []
    pairs_used = set()

    for i, c1 in enumerate(sphere_centres):
        for j, c2 in enumerate(sphere_centres):
            if i >= j:
                continue  # deduplicate — only process each pair once

            pair_key = (min(i, j), max(i, j))
            if pair_key in pairs_used:
                continue

            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist > neighbour_radius_mm:
                continue

            profile = get_dose_profile_between_points(
                dose_array, dose_sitk, c1, c2, n_samples=n_samples
            )
            profiles.append(profile)
            pairs_used.add(pair_key)
            # print(f"[INFO] Pair ({i},{j}) — distance: {dist:.1f} mm")

    print(f"[INFO] Total profiles extracted: {len(profiles)}")
    return profiles


def plot_dose_profiles(profiles, neighbour_radius_mm, n_samples=100):
    """
    Plot mean +/- std of all sphere-to-sphere dose profiles.
    """
    if not profiles:
        print("[WARNING] No profiles to plot.")
        return

    profiles_arr = np.array(profiles)   # shape (n_pairs, n_samples)
    mean_profile = np.mean(profiles_arr, axis=0)
    std_profile  = np.std(profiles_arr,  axis=0)
    x = np.linspace(0, 1, n_samples)

    peak = np.max(mean_profile)
    valley = np.min(mean_profile)

    upper_thresh = 0.75 * (peak-valley) + valley
    lower_thresh = 0.25 * (peak-valley) + valley

    # Find crossings
    upper_idx = np.where(mean_profile >= upper_thresh)[0]
    lower_idx = np.where(mean_profile >= lower_thresh)[0]

    fig, ax = plt.subplots(figsize=(9, 5))

    if len(upper_idx):
        left_upper = x[upper_idx[0]]
        right_upper = x[upper_idx[-1]]

        ax.axvline(left_upper, color='red', linestyle='--',
                   label='75% peak boundary')
        ax.axvline(right_upper, color='red', linestyle='--')

    if len(lower_idx):
        left_lower = x[lower_idx[0]]
        right_lower = x[lower_idx[-1]]

        ax.axvline(left_lower, color='green', linestyle=':')
        ax.axvline(right_lower, color='green', linestyle=':')

    upper_crossings = np.where(
        np.diff(np.sign(mean_profile - upper_thresh))
    )[0]

    for idx in upper_crossings:
        x_cross = np.interp(
            upper_thresh,
            [mean_profile[idx], mean_profile[idx + 1]],
            [x[idx], x[idx + 1]]
        )
        ax.axvline(x=x_cross, linestyle="--", alpha=0.8)
        print('75% threshold_distance', x_cross*42)

    # Crossings with lower threshold
    lower_crossings = np.where(
        np.diff(np.sign(mean_profile - lower_thresh))
    )[0]

    for idx in lower_crossings:
        x_cross = np.interp(
            lower_thresh,
            [mean_profile[idx], mean_profile[idx + 1]],
            [x[idx], x[idx + 1]]
        )
        ax.axvline(x=x_cross, linestyle=":", alpha=0.8)
        print('25% threshold_distance', x_cross * 42)

    ax.plot(x, mean_profile, color="royalblue", linewidth=2, label="Mean dose profile")
    ax.fill_between(
        x,
        mean_profile - std_profile,
        mean_profile + std_profile,
        alpha=0.3, color="royalblue", label="± 1 SD"
    )

    ax.axhline(upper_thresh, color='red', linestyle='--')
    ax.axhline(lower_thresh, color='green', linestyle=':')
    ax.set_xlabel("Fractional distance between sphere centres (0 → 1)")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title(
        f"Sphere-to-sphere dose profiles\n"
        f"(neighbour radius: {neighbour_radius_mm} mm, n={len(profiles)} pairs)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return profiles_arr, mean_profile, std_profile


def visualise_max_dose_slice(dose_array, masks_on_dose, roi_names, dose_sitk):
    """
    Find the slice containing the max dose voxel and plot:
    - dose heatmap
    - overlaid contours for each ROI
    """
    # ── Find slice with max dose ──────────────────────────────────────────
    max_idx = np.unravel_index(np.argmax(dose_array), dose_array.shape)
    z_slice = max_idx[0]  # z is axis 0 (slice index)

    dose_slice = dose_array[z_slice, :, :]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    im = ax.imshow(
        dose_slice,
        cmap="jet",
        origin="lower",
        norm=Normalize(vmin=0, vmax=np.max(dose_array))
    )

    plt.colorbar(im, ax=ax, label="Dose (Gy)")

    # ── Overlay contours for each ROI ────────────────────────────────────
    colors = plt.cm.Set1.colors  # distinct colors per ROI
    legend_patches = []

    for i, (roi_name, mask_on_dose) in enumerate(masks_on_dose.items()):
        roi_slice = mask_on_dose[z_slice, :, :].astype(np.uint8)

        if np.sum(roi_slice) == 0:
            continue  # ROI not present on this slice

        color = colors[i % len(colors)]
        ax.contour(roi_slice, levels=[0.5], colors=[color], linewidths=1.5)
        legend_patches.append(mpatches.Patch(color=color, label=roi_name))

    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    spacing = dose_sitk.GetSpacing()
    ax.set_title(
        f"Max dose slice (z={z_slice})  —  "
        f"max dose = {np.max(dose_array):.2f} Gy  |  "
        f"voxel spacing: {spacing[0]:.1f} x {spacing[1]:.1f} x {spacing[2]:.1f} mm"
    )
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")

    plt.tight_layout()
    plt.show()

def dilate_mask_mm(mask_sitk, radius_mm):
    """
    Morphologically dilate a binary SimpleITK mask by radius_mm in physical space.
    """
    # Convert radius from mm to voxels (use minimum spacing to be safe)
    spacing = mask_sitk.GetSpacing()
    radius_voxels = int(np.ceil(radius_mm / min(spacing)))

    dilated = sitk.BinaryDilate(
        mask_sitk,
        kernelRadius=(radius_voxels, radius_voxels, radius_voxels),
        kernelType=sitk.sitkBall,
        foregroundValue=1
    )
    return dilated

def rtstruct_to_masks(roi_names_to_process, rtstruct_path, reference_folder):
    """
    Build binary SimpleITK masks from RTSTRUCT contours,
    natively registered to the CT physical space.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(reference_folder)
    reader.SetFileNames(dicom_names)
    reference_image = reader.Execute()
    ref_size = reference_image.GetSize()   # (x, y, z)

    rtstruct = pydicom.dcmread(rtstruct_path)
    masks = {}

    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name   = roi.ROIName
        if roi_name not in roi_names_to_process:
            continue

        roi_contour = next(
            (rc for rc in rtstruct.ROIContourSequence
             if rc.ReferencedROINumber == roi_number), None
        )
        if roi_contour is None:
            continue

        mask_3d = np.zeros((ref_size[2], ref_size[1], ref_size[0]), dtype=np.uint8)

        for contour in roi_contour.ContourSequence:
            points    = np.array(contour.ContourData).reshape(-1, 3)
            idx_pts   = [reference_image.TransformPhysicalPointToIndex(tuple(p))
                         for p in points]
            z         = idx_pts[0][2]
            poly_pts  = np.array([[p[0], p[1]] for p in idx_pts], dtype=np.int32)

            slice_mask = np.zeros((ref_size[1], ref_size[0]), dtype=np.uint8)
            cv2.fillPoly(slice_mask, [poly_pts], 1)
            mask_3d[z, :, :] |= slice_mask

        mask_sitk = sitk.GetImageFromArray(mask_3d)
        mask_sitk.CopyInformation(reference_image)   #  real spacing/origin/direction
        masks[roi_name] = mask_sitk

    return masks


def load_dose_as_sitk(rt_dose_path):
    """
    Load RTDose and build a SimpleITK image with correct
    physical geometry (origin, spacing, direction) from DICOM tags.
    Returns (dose_sitk, dose_array_Gy).
    """
    dcm = pydicom.dcmread(rt_dose_path)

    dose_array = dcm.pixel_array.astype(np.float32) * float(dcm.DoseGridScaling)

    dose_sitk = sitk.GetImageFromArray(dose_array)

    # ── Spacing ──────────────────────────────────────────────────────────
    row_spacing = float(dcm.PixelSpacing[0])
    col_spacing = float(dcm.PixelSpacing[1])
    offsets     = [float(v) for v in dcm.GridFrameOffsetVector]
    # slice spacing from consecutive frame offsets
    if len(offsets) > 1:
        slice_spacing = abs(offsets[1] - offsets[0])
    else:
        slice_spacing = 1.0
    dose_sitk.SetSpacing((col_spacing, row_spacing, slice_spacing))

    # ── Origin ───────────────────────────────────────────────────────────
    # ImagePositionPatient gives the real-world position of voxel (0,0,0)
    ipp = [float(v) for v in dcm.ImagePositionPatient]
    dose_sitk.SetOrigin(tuple(ipp))

    # ── Direction cosines ────────────────────────────────────────────────
    iop = [float(v) for v in dcm.ImageOrientationPatient]
    row_cos  = iop[0:3]
    col_cos  = iop[3:6]
    # normal = cross product
    normal   = np.cross(row_cos, col_cos).tolist()
    direction = row_cos + col_cos + normal   # 9-element row-major
    dose_sitk.SetDirection(direction)

    return dose_sitk, dose_array


def resample_mask_to_dose(mask_sitk, dose_sitk):
    """
    Resample binary mask into the dose grid's physical space.
    """
    return sitk.Resample(
        mask_sitk,
        dose_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_sitk.GetPixelID()
    )

def load_dose(rt_dose_path):
    dose = pydicom.dcmread(rt_dose_path)

    dose_grid = dose.pixel_array.astype(np.float32)
    dose_grid *= float(dose.DoseGridScaling)

    return dose, dose_grid

def get_roi_mask(rtstruct_path, ct_folder, roi_name):
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=ct_folder,
        rt_struct_path=rtstruct_path
    )

    mask = rtstruct.get_roi_mask_by_name(roi_name)
    return mask



def compute_roi_stats(dose_grid, roi_mask):
    roi_dose = dose_grid[roi_mask]

    if roi_dose.size == 0:
        return None

    stats = {
        "mean": float(np.mean(roi_dose)),
        "max": float(np.max(roi_dose)),
        "min": float(np.min(roi_dose)),
        "median": float(np.median(roi_dose)),
        "D95": float(np.percentile(roi_dose, 5)),
        "D5": float(np.percentile(roi_dose, 95)),
        "volume_voxels": int(roi_dose.size),
    }

    return stats

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


        layout.addWidget(self.label)
        layout.addWidget(self.list_widget)
        layout.addLayout(button_layout)

        self.setLayout(layout)

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





# CT_folder_path = QFileDialog.getExistingDirectory(None, "Select CT Folder")
# rtstruct_path = QFileDialog.getOpenFileName(None, "Select RTStruct DICOM File", "", "DICOM Files (*.dcm)")[0]
# rt_dose_path = QFileDialog.getOpenFileName(None, "Select RTDose DICOM File", "", "DICOM Files (*.dcm)")[0]



# CT_folder_path = (r"C:\Users\MULCAHY\Desktop\2341684~BOISNARD SYLVETTE\2024-06__Studies\BOISNARD^SYLVETTE_2341684_CT_2024-06-21_120304_RT.THORAX(Adulte)_THORAX..2mm.HD.FoV_n209__00000")
# rtstruct_path = (r"C:\Users\MULCAHY\Desktop\2341684~BOISNARD SYLVETTE\2024-06__Studies\BOISNARD^SYLVETTE_2341684_RTst_2024-06-21_120304_RT.THORAX(Adulte)_Contours.lattice.Ok_n1__00000\Boisnard_rtdose_structures.dcm")
# rt_dose_path = (r"C:\Users\MULCAHY\Desktop\2341684~BOISNARD SYLVETTE\2024-06__Studies\RTdose_boisnard\DCMTPS_Calculated(BOISNARD SYLVETTE).dcm")

app    = QApplication(sys.argv)

CT_folder_path = select_CT_folder()
if not CT_folder_path:
    sys.exit()

rtstruct_path = select_RTStruct_file()
if not rtstruct_path:
    sys.exit()

rt_dose_path = select_RTDose_file()
if not rt_dose_path:
    sys.exit()

picker = ROIPicker(rtstruct_path)
picker.show()
app.exec_()

if picker.user_cancelled or not picker.selected_roi:
    sys.exit()



dose_sitk, dose_array = load_dose_as_sitk(rt_dose_path)

print("Dose max (Gy)      :", np.max(dose_array))
print("Dose > 5 Gy voxels :", np.sum(dose_array > 5))

masks = rtstruct_to_masks(picker.selected_roi, rtstruct_path, CT_folder_path)

results = {}
masks_on_dose = {}

SPHERE_ROI_NAME = 'CiblesExt_Lattice'   # adjust to match your actual ROI name
DILATION_MM = 5

for roi_name, mask_sitk in masks.items():
    if roi_name == SPHERE_ROI_NAME:
        mask_sitk = dilate_mask_mm(mask_sitk, DILATION_MM)
        print(f"[INFO] Dilated {roi_name} by {DILATION_MM} mm")

    mask_on_dose_sitk = resample_mask_to_dose(mask_sitk, dose_sitk)
    mask_on_dose      = sitk.GetArrayFromImage(mask_on_dose_sitk).astype(bool)
    masks_on_dose[roi_name] = mask_on_dose  # ← store it

    mask_on_dose_sitk = resample_mask_to_dose(mask_sitk, dose_sitk)
    mask_on_dose      = sitk.GetArrayFromImage(mask_on_dose_sitk).astype(bool)

    n_voxels = int(np.sum(mask_on_dose))
    if n_voxels == 0:
        print(f"[WARNING] No overlap after resampling for ROI: {roi_name}")
        results[roi_name] = None
        continue

    roi_dose = dose_array[mask_on_dose]

    voxel_spacing = dose_sitk.GetSpacing()  # (x, y, z) in mm
    voxel_volume_cc = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000  # mm³ → cc

    results[roi_name] = {
        "mean": float(np.mean(roi_dose)),
        "max": float(np.max(roi_dose)),
        "min": float(np.min(roi_dose)),
        "std": float(np.std(roi_dose)),
        "D95": float(np.percentile(roi_dose, 5)),
        "D98": float(np.percentile(roi_dose, 2)),
        "D50": float(np.percentile(roi_dose, 50)),
        "D5": float(np.percentile(roi_dose, 95)),
        "D2": float(np.percentile(roi_dose, 98)),
        "voxels": n_voxels,
        "volume_cc": round(n_voxels * voxel_volume_cc, 2)  # in cc/cm³
    }

print(results)

pvdr = results['CiblesExt_Lattice']["mean"] / results['GTV-cibleExt5mm']["mean"]
print(f"PVDR (mean peak / mean valley): {pvdr:.2f}")

NEIGHBOUR_RADIUS   = 45     # mm — change to 60 for enlarged lattice

sphere_centres = get_sphere_centres(rtstruct_path, SPHERE_ROI_NAME, CT_folder_path)

profiles = compute_sphere_profiles(
    dose_array, dose_sitk, sphere_centres,
    neighbour_radius_mm=NEIGHBOUR_RADIUS
)

print(f"Number of sphere centres found: {len(sphere_centres)}")
# for i, c in enumerate(sphere_centres):
#     print(f"  Sphere {i}: {c}")

print(f"Number of profiles extracted: {len(profiles)}")

profiles_arr, mean_profile, std_profile = plot_dose_profiles(
    profiles, neighbour_radius_mm=NEIGHBOUR_RADIUS
)

# visualise_max_dose_slice(dose_array, masks_on_dose, picker.selected_roi, dose_sitk)

# print("dose max:", np.max(dose_array))
# print("dose > 5Gy voxels:", np.sum(dose_array > 5))
# print("lattice overlap voxels:", np.sum(lattice_mask))
# print("gtv overlap voxels:", np.sum(gtv_mask))