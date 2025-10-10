"""SHiP NC magnet get map template.
   ==========

   Compute the field map for the NC magnets.
"""
import os, shutil
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from time import time
from scipy.interpolate import griddata
import pandas as pd
from scipy.spatial import cKDTree
import gzip, pickle
#import roxie_evaluator
import snoopy
import multiprocessing as mp
import h5py

SC_Ymgap = 15
RESOL_DEF = (2,2,5)
def get_fixed_params(yoke_type = 'Mag1'):
    SC = (yoke_type == 'Mag2')
    return {
    'yoke_type': yoke_type,
    'coil_material': 'hts_pencake.json' if SC else 'copper_water_cooled.json',
    'max_turns': 12 if SC else 10,
    'J_tar(A/mm2)': 583 if SC else -1,
    'coil_diam(mm)': 40 if SC else 9,
    'insulation(mm)': 1 if SC else 0.5,
    'winding_radius(mm)': 200 if SC else 0,
    'yoke_spacer(mm)': 5,
    'material': 'aisi1010.json',
    'field_density': 5,
    'delta_x(m)': 1 if SC else 0.5,
    'delta_y(m)': 1 if SC else 0.5,
    'delta_z(m)': 1 if SC else 0.5}

def get_magnet_params(params, 
                     Ymgap:float = 0.15,
                     yoke_type:str = 'Mag1',
                     resol = RESOL_DEF,
                     use_B_goal:bool = False,
                     materials_directory = None,
                     save_dir = None,
                     use_diluted = False):
    if len(params) == 15: params = params[1:]

    ratio_yoke_1 = params[7]
    ratio_yoke_2 = params[8]
    B_NI = params[13]
    params = params / 100
    Xmgap_1 = params[11]
    Xmgap_2 = params[12]
    d = get_fixed_params(yoke_type)
    d.update({
    'resol_x(m)': resol[0] / 100,
    'resol_y(m)': resol[1] / 100,
    'resol_z(m)': resol[2] / 100,
    'Z_pos(m)': -1*params[0],
    'Xmgap1(m)': Xmgap_1,
    'Xmgap2(m)': Xmgap_2,
    'Z_len(m)': 2 * params[0],
    'Xcore1(m)': params[1] + Xmgap_1,
    'Xvoid1(m)': params[1] + params[5] + Xmgap_2,
    'Xyoke1(m)': params[1] + params[5] + ratio_yoke_1 * params[1] + Xmgap_1,
    'Xcore2(m)': params[2] + Xmgap_2,
    'Xvoid2(m)': params[2] + params[6] + Xmgap_2,
    'Xyoke2(m)': params[2] + params[6] + ratio_yoke_2 * params[2] + Xmgap_2,
    'Ycore1(m)': params[3],
    'Yvoid1(m)': params[3] + Ymgap,
    'Yyoke1(m)': params[3] + params[9] + Ymgap,
    'Ycore2(m)': params[4],
    'Yvoid2(m)': params[4] + Ymgap,
    'Yyoke2(m)': params[4] + params[10] + Ymgap
    })
    if use_B_goal:
        if materials_directory is None:
            materials_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/materials')
        d['NI(A)'] = snoopy.get_NI(abs(B_NI), pd.DataFrame([d]),0, materials_directory = materials_directory)[0].item()
        if (B_NI > 0 and d['yoke_type'] == 'Mag3') or (B_NI < 0 and d['yoke_type'] == 'Mag1'):
            d['NI(A)'] = -d['NI(A)']
    elif use_diluted: d['NI(A)'] = B_NI
    else: d['NI(A)'] = abs(B_NI)

    if use_diluted and d['yoke_type'] == 'Mag3': 
        d['yoke_type'] = 'Mag1'

    if save_dir is not None:
        from csv import DictWriter
        with open(save_dir/"parameters.csv", "w", newline="") as f:
            w = DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)
    return d


def get_symmetry(points:np.array, B:np.array, reorder:bool = True):
   '''Applies symmetry to the computed magnetic field.'''
   points_1 = points
   points_2 = np.array([-points[:,0], points[:,1], points[:,2]]).T
   points_3 = np.array([-points[:,0], -points[:,1], points[:,2]]).T
   points_4 = np.array([points[:,0], -points[:,1], points[:,2]]).T
   B_1 = B
   B_2 = np.array([-B[:,0], B[:,1], B[:,2]]).T
   B_3 = np.array([B[:,0], B[:,1], -B[:,2]]).T
   B_4 = np.array([-B[:,0], B[:,1], -B[:,2]]).T
   points = np.vstack((points_1, points_2, points_3, points_4))
   B = np.vstack((B_1, B_2, B_3, B_4))
   if reorder:
      sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
      points = points[sorted_indices]
      B = B[sorted_indices]
   return points, B

def construct_grid(limits, 
                   resol = RESOL_DEF,
                   eps:float = 1e-12):
    '''Constructs a grid based on the limits and resolution given.'''
    (min_x, min_y, min_z), (max_x, max_y, max_z) = limits
    r_x, r_y, r_z = resol
    nx = int(round((max_x - min_x) / r_x)) + 1
    ny = int(round((max_y - min_y) / r_y)) + 1
    nz = int(round((max_z - min_z) / r_z)) + 1
    X = np.linspace(min_x, max_x, nx)
    Y = np.linspace(min_y, max_y, ny)
    Z = np.linspace(min_z, max_z, nz)
    X, Y, Z = np.meshgrid(X, Y, Z)
    return X / 100, Y / 100, Z / 100

def get_grid_data(points: np.array, B: np.array, new_points: tuple):
    '''Interpolates the magnetic field data to a new grid.'''
    t1 = time()
    '''Bx_out = griddata(points, B[:, 0], new_points, method='nearest', fill_value=0.0).ravel()
    By_out = griddata(points, B[:, 1], new_points, method='nearest', fill_value=0.0).ravel()
    Bz_out = griddata(points, B[:, 2], new_points, method='nearest', fill_value=0.0).ravel()'''
    new_points = np.column_stack((new_points[0].ravel(), new_points[1].ravel(), new_points[2].ravel()))
    Bx_out, By_out, Bz_out = np.zeros_like(new_points).T
    hull =  (new_points[:, 0] <= points[:, 0].max()) & \
            (new_points[:, 1] <= points[:, 1].max()) & \
            (new_points[:, 2] >= points[:, 2].min()) & (new_points[:, 2] <= points[:, 2].max())
    tree = cKDTree(points)
    _, idx = tree.query(new_points[hull], k=1)
    Bx_out[hull] = B[idx, 0]
    By_out[hull] = B[idx, 1]
    Bz_out[hull] = B[idx, 2]
    new_B = np.column_stack((Bx_out, By_out, Bz_out))
    print('Griddind / Interpolation time = {} sec'.format(time() - t1))
    return new_points, new_B

def get_vector_field(magn_params, materials_dir, use_diluted=False):
    if 'Mag2' in magn_params['yoke_type']:
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_ncsc(magn_params, 0, materials_directory=materials_dir)
    elif magn_params['yoke_type'][0] == 'Mag1':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(magn_params, 0, materials_directory=materials_dir, use_diluted_steel=use_diluted)
    elif magn_params['yoke_type'][0] == 'Mag3':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(magn_params, 0, materials_directory=materials_dir, use_diluted_steel=use_diluted)
    else: raise ValueError(f'Invalid yoke type - Received yoke_type {magn_params["yoke_type"][0]}')
    return points.round(4).astype(np.float16), B.round(4).astype(np.float16), M_i, M_c, Q, J

def run_fem(magn_params:dict,
            materials_dir = None, use_diluted = False):
    """Runs the finite element method to compute the magnetic field.
    Parameters:
    magn_params (dict): Dictionary containing the magnets parameters.
    materials_dir (str, optional): Directory containing the materials. Defaults is None, returning the data dir in tha parent dir.
    Returns:
    dict: A dictionary containing the position points and the computed magnetic field 'B'.
    """
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/materials')
    start = time()
    points, B, M_i, M_c, Q, J = get_vector_field(magn_params, materials_dir, use_diluted=use_diluted)
    C_i, C_c, C_edf = snoopy.compute_prices(magn_params, 0, M_i, M_c, Q, materials_directory = materials_dir)
    cost = C_i + C_c + C_edf
    end = time()
    print('FEM Computation time = {} sec'.format(end - start))
    return {'points':points, 'B':B}

def simulate_and_grid(params, points, use_diluted = False):
    return get_grid_data(**run_fem(params, use_diluted = use_diluted), new_points=points)[1]

def run(magn_params:dict,
        d_space = (((0.,4.), (0.,4.), (-1, 30.))),
        save_results:bool = False,
        output_file:str = './outputs',
        apply_symmetry:bool = False,
        cores:int = 1,
        use_diluted =  False
        ):
    """Simulates the magnetic field based on given parameters and performs various operations such as applying symmetry,
    plotting results, and saving results.
    Parameters:
    magn_params (dict): Dictionary containing magnetic parameters.
    output_file (str, optional): Directory to save outputs. Defaults to './outputs'.
    apply_symmetry (bool, optional): Whether to apply symmetry to the computed magnetic field. Defaults to False.
    plot_results (bool, optional): Whether to plot the results. Defaults to False.
    save_results (bool, optional): Whether to save the results to a file. Defaults to False.
    d_space (tuple, optional): Dimensions of the space returned. Since the problem is symmetric, it must be in the form (dx,dy,(-z_i,z_f)).
    Defaults to ((3.5, 4.5, (-15., 15.))).
    Returns:
    dict: A dictionary containing the computed points and magnetic field 'B'.
    """
    
    n_magnets = len(magn_params['yoke_type'])
    print('Starting simulation for {} magnets'.format(n_magnets))
    limits_quadrant = ((d_space[0][0], d_space[1][0], d_space[2][0]), (d_space[0][1],d_space[1][1], d_space[2][1]))
    resol = RESOL_DEF
    points = construct_grid(limits=limits_quadrant, resol=resol)
    params_split = [({k: [v[i]] for k, v in magn_params.items()}, points, use_diluted) for i in range(0, n_magnets)]
    if n_magnets>1:
        if params_split[1][0]['yoke_type'][0] == 'Mag2':
            for k in magn_params.keys():
                params_split[0][0][k] += params_split[1][0][k]
            params_split.pop(1)
    with mp.Pool(cores) as pool:
      B = pool.starmap(simulate_and_grid, params_split)
    B = np.sum(B, axis=0)

    points = np.column_stack([points[i].ravel() for i in range(3)])
    if apply_symmetry:
        points,B = get_symmetry(points, B, reorder = True)

    if save_results:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump({'points':points, 'B':B}, f)
        print('Results saved to', output_file)
    return {'points':points, 'B':B}
    
def simulate_field(params,
              Z_init = 0,
              fSC_mag:bool = True,
              d_space = ((4., 4., (-1, 30.))), 
              resol = RESOL_DEF,
              NI_from_B_goal:bool = True,
              file_name = 'data/outputs/fields.pkl',
              cores = 1,
              use_diluted = False):
    
    '''Simulates the magnetic field for the given parameters. If save_fields is True, the fields are saved to data/outputs/fields.pkl'''
    t1 = time()
    all_params = pd.DataFrame()
    Z_pos = 0.
    for i, mag_params in enumerate(params):
        Z_pos += mag_params[0]/100
        if mag_params[1]<1: continue
        if mag_params[2]<1: 
            Z_pos += 2 * mag_params[1]/100
            continue
        if fSC_mag and mag_params[14]>1e6:
            Ymgap = SC_Ymgap; yoke_type = 'Mag2'
        elif use_diluted: Ymgap = 0.; yoke_type = 'Mag1'
        else: 
            Ymgap = 0.; 
            yoke_type = 'Mag3' if mag_params[14]<0 else 'Mag1'
        p = get_magnet_params(mag_params, Ymgap=Ymgap, use_B_goal=NI_from_B_goal, yoke_type=yoke_type, resol = resol, use_diluted = use_diluted)
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)']
    fields = run(all_params.to_dict(orient='list'), d_space=d_space, apply_symmetry=False, cores=cores, use_diluted = use_diluted)
    fields['points'][:,2] += Z_init / 100
    print('Magnetic field simulation took', time()-t1, 'seconds')

    if file_name is not None:
        all_params.to_csv(os.path.join(os.path.dirname(file_name), 'magnet_params.csv'), index=False)
        time_str = time()
        with h5py.File(file_name, "w") as f:
            if '_mm' in file_name: f.create_dataset("points", data=fields['points'].astype(np.float16), compression=None)
            f.create_dataset("B", data=fields['B'].astype(np.float16), compression=None)
            d_space = ((d_space[0][0],d_space[0][1], RESOL_DEF[0]),(d_space[1][0],d_space[1][1], RESOL_DEF[1]),(d_space[2][0],d_space[2][1], RESOL_DEF[2]))
            f.create_dataset("d_space", data=np.array(d_space, dtype=np.int16), compression=None)
        print('Fields saved to', file_name)
        print('Saving took', time() - time_str, 'seconds')
    return fields
   


if __name__ == '__main__':
   
   MAIN_DIR = '/home/hep/lprate/projects/roxie_ship'
   parameters_filename = "/disk/users/gfrise/New_project/MuonsAndMatter/data/magnet_params.csv"
   output_file = "/disk/users/gfrise/New_project/MuonsAndMatter/data/outputs/fields_mm.npy"

   import pandas as pd
   magn_params = pd.read_csv(parameters_filename).to_dict(orient='list')
   print(magn_params)
   t1 = time()
   d = run(magn_params,output_file = output_file, save_results=True, cores = 7, use_diluted = False)
   print('total_time: ', time() - t1, ' sec')
   points = d['points']
   print('limits: ', points.min(axis=0), points.max(axis=0))
   print('shape: ', points.shape)