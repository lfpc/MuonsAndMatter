import numpy as np
import torch

def get_corners_from_detector(detector):

    def expand_corners(corners, dz, z_center):
        corners = np.array(corners)
        z_min = z_center - dz
        z_max = z_center + dz
        corners = corners.reshape(8, 2)
        z = np.full((8, 1), z_min)
        z[4:] = z_max
        corners = np.hstack([corners, z])
        return corners

    all_corners = []
    for magnet in detector['magnets']:
        for component in magnet['components']:
            corners = component['corners']
            dz = component['dz']
            z_center = component['z_center'] 
            corners = expand_corners(corners, dz, z_center)
            all_corners.append(corners)
    return torch.from_numpy(np.array(all_corners))

def generate_magnet_corners(params, fSC_mag=False, Z_GAP=10.0, SC_Ymgap=15.0, device=None):
    """
    Generate 3D corners for magnets from parameter vector using PyTorch.
    
    Args:
        params: Vector of magnet parameters (flattened) - can be list, numpy array, or torch tensor
        fSC_mag: Flag for superconducting magnets
        Z_GAP: Gap between magnets in cm
        SC_Ymgap: Y gap for superconducting magnets in cm
        N_PARAMS: Number of parameters per magnet
        device: PyTorch device (cuda/cpu). If None, uses params device or cpu
    
    Returns:
        torch tensor with shape [N, 8, 3] where N is total number of components
        Each component has 8 corners with (x, y, z) coordinates in meters
    """
    
    # Convert to torch tensor if needed
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32)
    
    if device is not None:
        params = params.to(device)
    N_PARAMS = params.size(-1)
    
    n_magnets = len(params)
    params = params.reshape(n_magnets, N_PARAMS)
    
    cm = 1.0
    Z = 0.0
    all_corners = []
    
    for nM in range(n_magnets):
        magnet = params[nM]
        
        zgap = magnet[0]            # zgap
        dZ = magnet[1].item()
        dXIn = magnet[2]            # dX
        dXOut = magnet[3]           # dX2  
        dYIn = magnet[4]            # dY
        dYOut = magnet[5]           # dY2
        gapIn = magnet[6]           # gap
        gapOut = magnet[7]          # gap2
        ratio_yokesIn = magnet[8]   # ratio_yoke_1
        ratio_yokesOut = magnet[9]  # ratio_yoke_2
        dY_yokeIn = magnet[10]      # dY_yoke_1
        dY_yokeOut = magnet[11]     # dY_yoke_2
        midGapIn = magnet[12]       # middleGap
        midGapOut = magnet[13]      # middleGap2
        NI = magnet[14]
        
        if dZ < 1 or dXIn < 1:
            continue
            
        # Check if superconducting
        is_SC = (abs(NI.item()) > 1E6) and fSC_mag
        Ymgap = SC_Ymgap * cm if is_SC else 0.0
        
        # Adjust dY for Ymgap (same as original code)
        dY = dYIn + Ymgap
        dY2 = dYOut + Ymgap
        
        # Anti-overlap parameter
        anti_overlap = 0.0
        
        # Generate 2D corners for all components (following original logic)
        corners_2d_list = []
        
        # Main Left magnet
        corners_2d_list.append(torch.tensor([
            midGapIn, 
            -(dY + dY_yokeIn) - anti_overlap, 
            midGapIn, 
            dY + dY_yokeIn - anti_overlap,
            dXIn + midGapIn, 
            dY - anti_overlap, 
            dXIn + midGapIn,
            -(dY - anti_overlap),
            midGapOut,
            -(dY2 + dY_yokeOut - anti_overlap), 
            midGapOut, 
            dY2 + dY_yokeOut - anti_overlap,
            dXOut + midGapOut, 
            dY2 - anti_overlap, 
            dXOut + midGapOut,
            -(dY2 - anti_overlap)
        ], device=params.device))
        
        # Top Left magnet
        cornersTL = torch.tensor([
            midGapIn + dXIn, dY,
            midGapIn,
            dY + dY_yokeIn,
            dXIn + ratio_yokesIn * dXIn + midGapIn + gapIn,
            dY + dY_yokeIn,
            dXIn + midGapIn + gapIn,
            dY,
            midGapOut + dXOut,
            dY2,
            midGapOut,
            dY2 + dY_yokeOut,
            dXOut + ratio_yokesOut * dXOut + midGapOut + gapOut,
            dY2 + dY_yokeOut,
            dXOut + midGapOut + gapOut,
            dY2
        ], device=params.device)
        
        # Main Side Left magnet
        cornersMainSideL = torch.tensor([
            dXIn + midGapIn + gapIn,
            -dY, 
            dXIn + midGapIn + gapIn,
            dY, 
            dXIn + ratio_yokesIn * dXIn + midGapIn + gapIn, 
            dY + dY_yokeIn,
            dXIn + ratio_yokesIn * dXIn + midGapIn + gapIn, 
            -(dY + dY_yokeIn), 
            dXOut + midGapOut + gapOut,
            -dY2, 
            dXOut + midGapOut + gapOut, 
            dY2,
            dXOut + ratio_yokesOut * dXOut + midGapOut + gapOut, 
            dY2 + dY_yokeOut, 
            dXOut + ratio_yokesOut * dXOut + midGapOut + gapOut,
            -(dY2 + dY_yokeOut)
        ], device=params.device)
        
        # Generate symmetric components
        cornersMainR = -corners_2d_list[0]  # Main Right
        cornersMainSideR = -cornersMainSideL  # Main Side Right
        
        # Generate Top Right (mirrored from Top Left)
        cornersTR = torch.zeros(16, device=params.device)
        for i in range(8):
            j = (11 - i) % 8
            cornersTR[2 * j] = -cornersTL[2 * i]
            cornersTR[2 * j + 1] = cornersTL[2 * i + 1]
        
        # Generate Bottom Left and Bottom Right
        cornersBL = -cornersTR
        cornersBR = -cornersTL
        
        # Add all components to the list
        corners_2d_list.extend([
            cornersMainR,      # Main Right
            cornersMainSideL,  # Main Side Left  
            cornersMainSideR,  # Main Side Right
            cornersTL,         # Top Left
            cornersTR,         # Top Right
            cornersBL,         # Bottom Left
            cornersBR          # Bottom Right
        ])
        
        # Convert 2D corners to 3D
        z_center = Z / 100  # Convert to meters
        dz = dZ / 100 / 2   # Half-length in meters
        z0 = z_center - dz  # Bottom Z
        z1 = z_center + dz  # Top Z
        
        for corners_2d_flat in corners_2d_list:
            # Reshape to get (x,y) pairs
            corners_2d_pairs = corners_2d_flat.reshape(8, 2) / 100  # Convert to meters
            
            # Create 3D corners: first 4 points at z0, last 4 points at z1
            corners_3d = torch.zeros((8, 3), device=params.device)
            
            # Bottom face (z0) - first 4 points
            corners_3d[:4, :2] = corners_2d_pairs[:4]  # x, y
            corners_3d[:4, 2] = z0  # z
            
            # Top face (z1) - last 4 points  
            corners_3d[4:, :2] = corners_2d_pairs[4:]  # x, y
            corners_3d[4:, 2] = z1  # z
            
            all_corners.append(corners_3d)
        
        # Update Z position for next magnet
        Z += dZ + zgap
    
    return torch.stack(all_corners) if all_corners else torch.empty((0, 8, 3), device=params.device)


def get_blocks_hash(geo:torch.tensor,grid_dims = (100, 100, 1000),grid_lims = ((-5, 5), (-5, 5), (0, 40))):

    x_lim, y_lim, z_lim = grid_lims

    numel_grid = grid_dims[0] * grid_dims[1] * grid_dims[2]
    grid_size_x = (x_lim[1] - x_lim[0]) / grid_dims[0]
    grid_size_y = (y_lim[1] - y_lim[0]) / grid_dims[1]
    grid_size_z = (z_lim[1] - z_lim[0]) / grid_dims[2]

    min_x = geo[:,:,0].min(-1).values - grid_size_x
    max_x = geo[:,:,0].max(-1).values + grid_size_x
    min_y = geo[:,:,1].min(-1).values - grid_size_y
    max_y = geo[:,:,1].max(-1).values + grid_size_y
    min_z = geo[:,:,2].min(-1).values - grid_size_z
    max_z = geo[:,:,2].max(-1).values + grid_size_z

    a_positions = torch.meshgrid(
    torch.linspace(x_lim[0], x_lim[1], grid_dims[0]),
    torch.linspace(y_lim[0], y_lim[1], grid_dims[1]),
    torch.linspace(z_lim[0], z_lim[1], grid_dims[2]),
    indexing='ij'
    )

    voxel_indices = []
    block_ids = []

    for i in range(len(geo)):
        candidate_mask = ((a_positions[0] >= min_x[i]) & (a_positions[0] <= max_x[i]) &
                        (a_positions[1] >= min_y[i]) & (a_positions[1] <= max_y[i]) &
                        (a_positions[2] >= min_z[i]) & (a_positions[2] <= max_z[i]))
        voxel_indices_for_block = torch.nonzero(candidate_mask.flatten(), as_tuple=True)[0]
        
        if len(voxel_indices_for_block) > 0:
            voxel_indices.append(voxel_indices_for_block)
            block_ids.append(torch.full_like(voxel_indices_for_block, fill_value=i))

    if len(voxel_indices) > 0:
        voxel_indices = torch.cat(voxel_indices)
        block_ids = torch.cat(block_ids)
    else:
        voxel_indices = torch.tensor([], dtype=torch.long)
        block_ids = torch.tensor([], dtype=torch.long)

    voxel_indices, sorted_indices = torch.sort(voxel_indices)
    block_ids = block_ids[sorted_indices]
    voxel_offsets = torch.zeros(numel_grid + 1, dtype=torch.int32)

    voxel_indices, counts = torch.unique_consecutive(voxel_indices, return_counts=True)
    voxel_offsets.scatter_add_(0, voxel_indices + 1, counts.to(torch.int32))
    voxel_offsets = torch.cumsum(voxel_offsets, dim=0)
    return voxel_offsets, block_ids


def get_cavern(detector):
    if 'cavern' not in detector:
        return torch.tensor([[-30, 30, -30, 30, 0], [-30, 30, -30, 30, 0]], dtype=torch.float32)
    cavern = detector['cavern']
    TCC8 = cavern[0]
    ECN3 = cavern[1]
    TCC8_params = [TCC8['x1'], TCC8['x2'], TCC8['y1'], TCC8['y2'], TCC8['z_center']+TCC8['dz']]
    ECN3_params = [ECN3['x1'], ECN3['x2'], ECN3['y1'], ECN3['y2'], ECN3['z_center']-ECN3['dz']]
    return torch.tensor([TCC8_params,ECN3_params], dtype=torch.float32)
