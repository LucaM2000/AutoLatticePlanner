import numpy as np

sph_radius = 7.5
x_sph_spacing = 60
y_sph_spacing = 30
z_sph_spacing = 30
xgrid = 540
ygrid = 540
zgrid = 540

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


def optimize_grid_orientation(grid_points, distance_map, target_mask, sphere_radius, tumor_center, spacing, target_weight=5.0):
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
            target_mask_check = batch_check_spheres_inside(positions, sphere_radius, target_mask, tolerance = 0.1)
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




def optimize_grid_orientation_and_translation(grid_points, tumor_center, distance_map, sphere_radius,spacing, target_mask=None, target_weight=5.0, translation_range=10, translation_step=1, num_rotations=1000):

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
            target_mask_check = batch_check_spheres_inside(final_voxel_positions, sphere_radius, target_mask, tolerance= 0.1)
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