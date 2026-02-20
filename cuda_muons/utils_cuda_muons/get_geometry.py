import numpy as np
import torch
import h5py


def expand_corners(corners, dz, z_center):
        corners = np.array(corners)
        z_min = z_center - dz
        z_max = z_center + dz
        corners = corners.reshape(8, 2)
        z = np.full((8, 1), z_min)
        z[4:] = z_max
        corners = np.hstack([corners, z])
        return corners
def get_corners_from_detector(detector, use_symmetry=True):
    all_corners = []
    for magnet in detector['magnets']:
        if use_symmetry: components = magnet['components'][:3]
        else: components = magnet['components']
        for component in components:
            corners = component['corners']
            dz = component['dz']
            z_center = component['z_center'] 
            corners = expand_corners(corners, dz, z_center)
            all_corners.append(corners)
    return torch.from_numpy(np.array(all_corners))

def get_cavern(detector):
    if 'cavern' not in detector:
        return torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32)
    cavern = detector['cavern']
    TCC8 = cavern[0]
    ECN3 = cavern[1]
    TCC8_params = [TCC8['x1'], TCC8['x2'], TCC8['y1'], TCC8['y2'], TCC8['z_center']+TCC8['dz']]
    ECN3_params = [ECN3['x1'], ECN3['x2'], ECN3['y1'], ECN3['y2'], ECN3['z_center']-ECN3['dz']]
    return torch.tensor([TCC8_params,ECN3_params], dtype=torch.float32)

def get_magnetic_field(detector):
    mag_dict = detector['global_field_map']
    if not mag_dict:
        return get_uniform_field(detector)
    if isinstance(mag_dict['B'], str):
        with h5py.File(mag_dict['B'], 'r') as f:
            mag_dict['B'] = f["B"][:]
    return mag_dict

def create_z_axis_grid(corners_tensor: torch.Tensor, sz: int) -> list[list[int]]:
    """
    Builds a Z-axis spatial grid and returns it in a flattened, GPU-friendly
    format (CRS) directly, along with the grid's metadata.

    This function performs the entire grid construction in a single, vectorized
    operation without creating intermediate Python list structures.

    Args:
        corners_tensor (torch.Tensor): A tensor of shape (N, 8, 3) containing the
                                       vertex coordinates for N ARB8 geometries.
        sz (int): The number of cells (slices) to divide the Z-axis into.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: A tuple containing:
            - cell_starts (torch.Tensor, int32): Start indices for each cell. Shape: (sz + 1,).
            - item_indices (torch.Tensor, int32): Flat array of all geometry indices, grouped by cell.
    """
    all_z_coords = corners_tensor[:, :, 2]
    z_min_global = all_z_coords.min()
    z_max_global = max(all_z_coords.max(),30.0)
    cell_boundaries = torch.linspace(z_min_global, z_max_global, sz + 1)
    cell_z_starts = cell_boundaries[:-1]
    cell_z_ends = cell_boundaries[1:]
    geom_z0 = corners_tensor[:, 0, 2]
    geom_z1 = corners_tensor[:, 7, 2]
    overlap_matrix = torch.logical_not((geom_z0.unsqueeze(0) > cell_z_ends.unsqueeze(1)).logical_or(geom_z1.unsqueeze(0) < cell_z_starts.unsqueeze(1)))
    cell_indices_flat, geom_indices_flat = torch.where(overlap_matrix)
    item_indices = geom_indices_flat.to(torch.int32)
    counts = torch.bincount(cell_indices_flat, minlength=sz)
    zero_prefix = torch.tensor([0], dtype=torch.int32)
    cell_starts = torch.cat((zero_prefix, counts.cumsum(dim=0))).to(torch.int32)
    return cell_starts, item_indices

def is_inside(points, corners):
    """
    Optimized version with AABB (Axis-Aligned Bounding Box) pre-filtering.
    
    points: (N_points, 3)
    corners: (1, 8, 3)  <-- Expecting single config per call as per your usage
    Returns: (N_points, 1) boolean array
    """

    points = np.asarray(points)
    if corners.ndim == 2:
        corners = corners[None, ...]
    c_min = corners.min(axis=(1, 2)) 
    c_max = corners.max(axis=(1, 2)) 
    in_box_mask = np.all((points >= c_min[0]) & (points <= c_max[0]), axis=1) 
    if not np.any(in_box_mask):
        return np.zeros((points.shape[0], 1), dtype=bool)
    candidates = points[in_box_mask] 

    face_indices = np.array([
        [0, 1, 2], [4, 5, 6], [0, 3, 7], 
        [1, 2, 6], [0, 4, 5], [3, 2, 6]
    ])

    p0 = corners[:, face_indices[:, 0]]  # (1, 6, 3)
    p1 = corners[:, face_indices[:, 1]]
    p2 = corners[:, face_indices[:, 2]]

    # Compute normals (Only done once per config, very fast)
    normals = np.cross(p1 - p0, p2 - p0, axis=-1)
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9)
    
    d = -np.sum(normals * p0, axis=-1)  # (1, 6)
    centroids = np.mean(corners, axis=1)  
    point_sides = (
        np.sum(candidates[:, None, None, :] * normals[None, :, :, :], axis=-1) 
        + d[None, :, :]
    )
    
    centroid_sides = (
        np.sum(centroids[None, :, None, :] * normals[None, :, :, :], axis=-1)
        + d[None, :, :]
    )
    
    # (K, 1, 6)
    is_outside_face = (point_sides * centroid_sides) < -1e-9
    candidate_inside = ~np.any(is_outside_face, axis=2) 
    final_mask = np.zeros((points.shape[0], 1), dtype=bool)
    final_mask[in_box_mask] = candidate_inside

    return final_mask

def get_uniform_field(detector, use_symmetry=True):
    from lib.magnet_simulations import RESOL_DEF, construct_grid
    import time
    resol = RESOL_DEF
    dx = detector['dx']*100  # in cm
    dy = detector['dy']*100  # in cm
    dz = detector['dz']*100  # in cm
    max_x = int((dx // resol[0]) * resol[0])
    max_y = int((dy // resol[1]) * resol[1])
    max_z = int(((dz+200) // resol[2]) * resol[2])
    limits_quadrant = ((0, 0, -50), (max_x+50,max_y+50, max_z))
    points = construct_grid(limits=limits_quadrant, resol=resol)
    points = np.column_stack([points[i].ravel() for i in range(3)])
    field_map = np.zeros_like(points)
    for magnet in detector['magnets']:
        if use_symmetry: components = magnet['components'][:3]
        else: components = magnet['components']
        for component in components:
            t0 = time.time()
            field_uni = np.array(component['field'])  # in Tesla
            corners = component['corners']
            dz = component['dz']
            z_center = component['z_center'] 
            mag_corners = expand_corners(corners, dz, z_center)
            t1 = time.time()
            print("TIME EXPAND:", t1 - t0)
            inside = is_inside(points, mag_corners.reshape(1,8,3)).reshape(-1)
            t2 = time.time()
            print("TIME INSIDE:", t2-t1)
            print("SHAPES:", inside.shape, field_map.shape, field_uni.shape)
            field_map[inside] += field_uni 
            print("ADDED FIELD FOR COMPONENT, took ", time.time()-t2)

    return {'B': field_map,
                'range_x': [limits_quadrant[0][0],limits_quadrant[1][0], RESOL_DEF[0]],
                'range_y': [limits_quadrant[0][1],limits_quadrant[1][1], RESOL_DEF[1]],
                'range_z': [limits_quadrant[0][2],limits_quadrant[1][2], RESOL_DEF[2]]}
