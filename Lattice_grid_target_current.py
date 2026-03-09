import gatetools as gt
from pathlib import Path
import glob
import os
import itk
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage import measure
import json
import matplotlib

def main():

    matplotlib.use('TkAgg')


    print("Loading image data...")

    with open("paths.json", "r") as f:
        data = json.load(f)

    dicom_series_path = data["dicom_series_path"]
    rtstruct_path = data["rtstruct_path"]
    ct_files = sorted(glob.glob(os.path.join(dicom_series_path, "*.dcm")))
    ct_image = gt.read_dicom(ct_files)
    additional_target = data.get("additional_target", False)

    lattice_geometry = data["lattice_geometry"]
    tolerance = data["Tolerance"]

    if lattice_geometry == 'Normal (Duriseti)':
        sph_radius = 7.5
        x_sph_spacing = 60
        y_sph_spacing = 30
        z_sph_spacing = 30

    else:
        sph_radius = 10
        x_sph_spacing = 80
        y_sph_spacing = 40
        z_sph_spacing = 40

    print('geometry selection test:', lattice_geometry, x_sph_spacing, y_sph_spacing, z_sph_spacing, sph_radius)

    ct_shape = itk.GetArrayFromImage(ct_image).shape  # [z, y, x]
    spacing = np.array(ct_image.GetSpacing())         # [x, y, z]
    origin = np.array(ct_image.GetOrigin())           # [x, y, z]
    direction = np.array(ct_image.GetDirection())

    print("Image data loaded!")

    print(dicom_series_path)

    print('Spacing:', spacing)

    # print('Origin:', origin)
    # print('Direction:', direction)

    ## Lattice Dimensions
    # sph_radius = 10
    # x_sph_spacing = 80
    # y_sph_spacing = 40
    # z_sph_spacing = 40
    xgrid = 540
    ygrid = 540
    zgrid = 540

    # Tolerance = percentage of sphere required inside volume to be counted (0.1 = 10% of sphere, 1 = whole sphere)


    def plot_sphere(ax, center, radius=0.5, resolution=10, color='b'):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot wireframe for the sphere with the specified color
        ax.plot_wireframe(x, y, z, color=color, alpha=0.3, rstride=1, cstride=1)


    def rotate_grid(grid_points, rotation_matrix, tumor_center):
        coords = grid_points[:, :3].astype(float)
        # colors = grid_points[:, 3].reshape(-1, 1)

        translated_points = coords - tumor_center
        rotated_points = np.dot(translated_points, rotation_matrix.T)
        final_points = rotated_points + tumor_center

        combined = np.hstack([final_points])

        return combined



    def batch_check_spheres_inside(rotated_points, radius, distance_map, tolerance):
        rotated_points = np.asarray(rotated_points)
        if rotated_points.ndim == 1:
            rotated_points = rotated_points.reshape(1, -1)

        rotated_points = np.array(rotated_points, dtype=float)
        indices = np.round(rotated_points).astype(int)
        x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]

        # Bounds check based on shape (x_dim, y_dim, z_dim)
        valid = (
            (x >= 0) & (x < distance_map.shape[0]) &
            (y >= 0) & (y < distance_map.shape[1]) &
            (z >= 0) & (z < distance_map.shape[2])
        )

        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = z[valid]

        # Correct axis order preserved: X=0, Y=1, Z=2
        distances = distance_map[x_valid, y_valid, z_valid]
        inside = distances >= radius * tolerance

        result_mask = np.zeros(rotated_points.shape[0], dtype=bool)
        result_mask[np.where(valid)[0][inside]] = True
        return result_mask


    def optimize_grid_orientation(grid_points, distance_map, target_mask, sphere_radius, tumor_center, spacing, target_weight=100.0):
        best_orientation = None
        best_fitness = 0
        best_inside_tumour = []
        best_inside_targets = []
        best_tumor_score = 0
        best_target_score = 0

        for _ in range(100000):
            rotation = R.random()
            rotation_matrix = rotation.as_matrix()

            physical_grid = grid_points[:, :3]
            rotated_grid_points = rotate_grid(physical_grid, rotation_matrix, tumor_center)
            positions = rotated_grid_points[:, :3] / spacing[::-1]

            # Always check coverage inside tumor
            tumor_mask = batch_check_spheres_inside(positions, sphere_radius, distance_map, tolerance = tolerance)
            tumor_score = np.count_nonzero(tumor_mask)

            # Check coverage inside additional target if provided
            if target_mask is not None:
                target_mask_check = batch_check_spheres_inside(positions, sphere_radius, target_mask, tolerance = 0.01)
                target_score = np.count_nonzero(target_mask_check)
                fitness = tumor_score + target_weight * target_score
            else:
                target_mask_check = np.zeros_like(tumor_mask, dtype=bool)
                target_score = 0
                fitness = tumor_score  # no weighting needed

            if fitness > best_fitness:
                best_fitness = fitness
                best_orientation = rotation_matrix
                best_inside_tumour = positions[tumor_mask]
                best_inside_targets = positions[target_mask_check]
                best_tumor_score = tumor_score
                best_target_score = target_score

        return (best_orientation, best_fitness, best_inside_tumour, best_inside_targets, best_tumor_score, best_target_score)




    def optimize_grid_orientation_and_translation(grid_points, tumor_center, distance_map, sphere_radius,spacing, target_mask=None, target_weight=100.0, translation_range=10, translation_step=1, num_rotations=1000):

        best_orientation = None
        best_translation = None
        best_fitness = 0
        best_inside_tumour = []
        best_inside_targets = []
        best_angles = []
        best_tumor_score = 0
        best_target_score = 0

        translation_offsets = np.arange(-translation_range, translation_range + translation_step, translation_step)

        for _ in range(num_rotations):
            physical_grid = grid_points[:, :3]
            colors = grid_points[:, 3] if grid_points.shape[1] > 3 else np.ones(grid_points.shape[0])

            rotation = R.random()
            rotation_matrix = rotation.as_matrix()
            rotated_grid_points = rotate_grid(physical_grid, rotation_matrix, tumor_center)
            current_grid = rotated_grid_points[:, :3]
            current_translation = [0, 0, 0]

            # Translation along each axis
            for i, axis in enumerate(['x', 'y', 'z']):
                best_d = 0
                best_count = 0
                for offset in translation_offsets:
                    translation_vector = np.zeros(3)
                    translation_vector[i] = offset
                    translated_physical = current_grid + translation_vector
                    translated_voxel = translated_physical / spacing[::-1]
                    mask = batch_check_spheres_inside(translated_voxel, sphere_radius, distance_map, tolerance = tolerance)
                    count = np.count_nonzero(mask)
                    if count > best_count:
                        best_count = count
                        best_d = offset
                current_translation[i] = best_d
                translation_vector = np.zeros(3)
                translation_vector[i] = best_d
                current_grid += translation_vector

            # Final fitness evaluation
            final_voxel_positions = current_grid / spacing[::-1]
            tumor_mask = batch_check_spheres_inside(final_voxel_positions, sphere_radius, distance_map, tolerance = tolerance)
            tumor_score = np.count_nonzero(tumor_mask)

            if target_mask is not None:
                target_mask_check = batch_check_spheres_inside(final_voxel_positions, sphere_radius, target_mask, tolerance= 0.01)
                target_score = np.count_nonzero(target_mask_check)
                fitness = tumor_score + target_weight * target_score
            else:
                target_mask_check = np.zeros_like(tumor_mask, dtype=bool)
                target_score = 0
                fitness = tumor_score

            if fitness > best_fitness:
                best_fitness = fitness
                best_orientation = rotation_matrix
                best_translation = tuple(current_translation)
                best_inside_tumour = final_voxel_positions[tumor_mask]
                best_inside_targets = final_voxel_positions[target_mask_check]
                best_angles = rotation.as_euler('zxy', degrees=True)
                best_tumor_score = tumor_score
                best_target_score = target_score

        return (best_orientation, best_translation, best_fitness, best_inside_tumour, best_inside_targets, best_angles, best_tumor_score, best_target_score)


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

    print("Generating grid....")

    gridx = []
    gridy = []
    gridz = []
    colors = []

    xno = int(np.round(xgrid / x_sph_spacing))
    yno = int(np.round(ygrid / y_sph_spacing))
    zno = int(np.round(zgrid / z_sph_spacing))

    for i in range(zno + 1):  # z layers
        for u in range(yno + 1):  # y rows
            for r in range(xno + 1):  # x cols
                # Start with base grid point
                gx = r * x_sph_spacing
                gy = u * y_sph_spacing
                gz = i * z_sph_spacing

                # Offset every second row in y (to form 2D hex pattern)
                if u % 2 == 1:
                    gx += x_sph_spacing / 2

                # Offset every second z-layer
                if i % 2 == 1:
                    gy += y_sph_spacing

                # Append only if within bounds
                if gx <= xgrid and gy <= ygrid and gz <= zgrid:
                    gridx.append(gx)
                    gridy.append(gy)
                    gridz.append(gz)

                    grid_color = 'b' if (r + u + i) % 2 == 0 else 'r'
                    colors.append(grid_color)

    # Create full 3D grid
    all_grid_points = np.array(list(zip(gridx, gridy, gridz)))

    # Center it
    grid_points = all_grid_points - np.mean(all_grid_points, axis=0)

    # Store in Nx4 array with dummy 4th column
    grid_points1 = np.zeros((len(grid_points), 4))
    grid_points1[:, :3] = grid_points



    # Optional: assign dummy value or use color index later
    grid_points1[:, 3] = [0 if c == 'b' else 1 for c in colors]

    # Check
    print(np.shape(grid_points1))
    print("Center of grid (should be ~0):", np.mean(grid_points1[:, :3], axis=0))

    print("Grid generated!")

    # mask = itk.imread("Lattice_target.mhd")
    # target_mask = itk.imread("Additional_target.mhd")
    # mask_spacing = np.array(mask.GetSpacing()) # x y z in mm
    # mask_numpy_array = itk.GetArrayFromImage(mask) #the coordinate system is [z, y, x]
    # spacing = mask_spacing




    if additional_target is True:
        target_volume = np.load("additional_target_volume.npy")
    else:
        target_volume = None

    # Running optimizer for separate GTVs
    mask_folder = Path("Masks")

    mask_files = sorted(mask_folder.glob("Lattice_target_*.mhd"))

    spheres_inside_array = []

    for mask_file in mask_files:
        roi_id = mask_file.stem.replace("Lattice_target_", "")

        distance_file = mask_folder / f"distance_map_{roi_id}.npy"
        volume_file = mask_folder / f"tumour_volume_{roi_id}.npy"

        mask = itk.imread(str(mask_file))
        distance_map = np.load(distance_file)
        tumour_volume = np.load(volume_file)



        verts, faces, normals, values = measure.marching_cubes(tumour_volume, level=0)



        centroid_x = np.mean(verts[:, 0])
        centroid_y = np.mean(verts[:, 1])
        centroid_z = np.mean(verts[:, 2])

        centroid = (centroid_x, centroid_y, centroid_z)

        centred_points = grid_points1[:, :3] + centroid

        colors = grid_points1[:, 3].reshape(-1, 1)
        centred_grid = np.hstack([centred_points, colors])

        print("Rotating grid...")

        # # Call the optimize_grid_orientation function and get the results
        (best_orientation,
         best_fitness_rotation,
         best_inside_rotation,
         best_inside_target_rotation,
         best_tumor_score_rotation,
         best_target_score_rotation) = optimize_grid_orientation(centred_grid, distance_map, target_volume,  sph_radius, centroid, spacing)

        # Now use the best_orientation to rotate the grid
        # optimized_grid_points = rotate_grid(centred_grid, best_orientation, centroid)

        print('Spheres inside based on rotation:', len(best_inside_rotation))
        print('Spheres covering target based on rotation:', len(best_inside_target_rotation))



        print('Translating & rotating grid...')

        (best_orientation,
        best_translation,
        best_fitness_rotation_translation,
        best_inside_tumour_RT,
        best_inside_targets_RT,
        best_angles_RT,
        best_tumor_score_RT,
        best_target_score_RT) = optimize_grid_orientation_and_translation(
            centred_grid, tumor_center=centroid, distance_map=distance_map, sphere_radius=sph_radius, spacing=spacing, target_mask = target_volume,
            translation_range=70, translation_step=2, num_rotations=5000)

        print('Spheres inside based on rotation & translation:', len(best_inside_tumour_RT))
        print('Spheres target coverage based on rotation & translation:', len(best_inside_targets_RT))

        if  (best_fitness_rotation) > (best_fitness_rotation_translation):
            spheres_inside = np.array(best_inside_rotation)
            spheres_inside_array.append(spheres_inside)
            chosen_method = "Rotation"
        else:
            spheres_inside = np.array(best_inside_tumour_RT)
            spheres_inside_array.append(spheres_inside)
            chosen_method = "Rotation + Translation"

        # if  len(best_inside_rotation) > len(best_inside_tumour_RT):
        #     spheres_inside = np.array(best_inside_rotation)
        #     chosen_method = "Rotation"
        # else:
        #     spheres_inside = np.array(best_inside_tumour_RT)
        #     chosen_method = "Rotation + Translation"

        print("Grid optimised!", f"Chosen method: {chosen_method}")

        if best_fitness_rotation > best_fitness_rotation_translation:
            best_fitness = len(best_inside_rotation)
            print("Number of spheres inside (rotation only):", best_fitness)

        else:
            best_fitness = len(best_inside_tumour_RT)
            print("Number of spheres inside (translation & rotation):", best_fitness)


    mask_array_1 = np.zeros(ct_shape, dtype=np.uint8)
    mask_array_2 = np.zeros(ct_shape, dtype=np.uint8)

    spheres_inside_array = np.vstack(spheres_inside_array)
    print(len(spheres_inside_array))

    print("Drawing spheres in mask...")

    for i, (x, y, z) in enumerate(spheres_inside_array):
        center_mm = np.array([float(x), float(y), float(z)])
        draw_sphere_mask(mask_array_1, center_mm, sph_radius, mask_spacing)

    for i, (x, y, z) in enumerate(spheres_inside_array):
        center_mm = np.array([float(x), float(y), float(z)])
        draw_sphere_mask(mask_array_2, center_mm, sph_radius + 5, mask_spacing)



    # Convert to ITK and save both
    def save_mask_as_mhd(mask_array, filename, spacing, origin, direction):
        itk_image = itk.image_from_array(mask_array.astype(np.uint8))
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.SetDirection(direction)
        itk.imwrite(itk_image, filename)

    save_mask_as_mhd(mask_array_1, "all_spheres_original_1.mhd", spacing, origin, direction)
    save_mask_as_mhd(mask_array_2, "all_spheres_expanded_1.mhd", spacing, origin, direction)

    np.savetxt("spheres_inside.txt", spheres_inside_array)


    print("Masks saved!")


    print("Number of Spheres in target volume:", best_fitness)

if __name__ == "__main__":
    main()

