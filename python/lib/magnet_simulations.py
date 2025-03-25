"""SHiP NC magnet get map template.
   ==========

   Compute the field map for the NC magnets.
"""
import os
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
from lib.reference_designs.params import new_parametrization


RESOL_DEF = (0.02,0.02,0.05)
def get_fixed_params(yoke_type = 'Mag1'):
    return {
    'yoke_type': yoke_type,
    'coil_material': 'hts_pencake.json' if yoke_type == 'Mag2' else 'copper_water_cooled.json',
    'max_turns': 12 if yoke_type == 'Mag2' else 10,
    'J_tar(A/mm2)': 320 if yoke_type == 'Mag2' else -1,
    'coil_diam(mm)': 20 if yoke_type == 'Mag2' else 9,
    'insulation(mm)': 8 if yoke_type == 'Mag2' else 0.5,
    'yoke_spacer(mm)': 5,
    'material': 'aisi1010.json',
    'field_density': 5,
    'delta_x(m)': 1 if yoke_type == 'Mag2' else 0.5,
    'delta_y(m)': 1 if yoke_type == 'Mag2' else 0.5,
    'delta_z(m)': 1 if yoke_type == 'Mag2' else 0.5}



def get_magnet_params(params, 
                     Ymgap:float = 0.05,
                     z_gap:float = 0.1,
                     yoke_type:str = 'Mag1',
                     resol = RESOL_DEF,
                     B_goal:float = None,
                     materials_directory = None,
                     save_dir = None):
    
    ratio_yoke_1 = params[7]
    ratio_yoke_2 = params[8]
    if B_goal is not None:
        NI = None
    else: NI = params[13]
    params /= 100
    Xmgap_1 = params[11]
    Xmgap_2 = params[12]
    d = get_fixed_params(yoke_type)
    d.update({
    'NI(A)': NI,
    'resol_x(m)': resol[0],
    'resol_y(m)': resol[1],
    'resol_z(m)': resol[2],
    'Z_pos(m)': -1*params[0],
    'Xmgap1(m)': Xmgap_1,
    'Xmgap2(m)': Xmgap_2,
    'Z_len(m)': 2 * params[0] - z_gap,
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
    if NI is None:
        if materials_directory is None:
            materials_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/materials')
        d['NI(A)'] = snoopy.get_NI(B_goal, pd.DataFrame([d]),0, materials_directory = materials_directory)[0]
    if save_dir is not None:
        from csv import DictWriter
        with open(save_dir/"parameters.csv", "w", newline="") as f:
            w = DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)
    return d

def get_melvin_params(params,
              fSC_mag:bool = False,
              z_gap = 0.1,
              resol = RESOL_DEF,
              NI_from_B_goal:bool = True):
    all_params = pd.DataFrame()
    Z_pos = 0.
    for i, (mag,idx) in enumerate(new_parametrization.items()):
        mag_params = params[idx]
        if mag_params[0]<1: continue
        if mag_params[1]<1: 
            Z_pos += 2 * mag_params[0]/100 - z_gap
            continue
        if mag == 'HA': Ymgap=0.; yoke_type = 'Mag1'; B_goal = 1.9 if NI_from_B_goal else None
        elif mag in ['M1', 'M2', 'M3']: Ymgap = 0.; B_goal = 1.9 if NI_from_B_goal else None; yoke_type = 'Mag1'
        else: Ymgap = 0.; B_goal = 1.9 if NI_from_B_goal else None; yoke_type = 'Mag3'
        if fSC_mag:
            if mag == 'M1': continue
            elif mag == 'M3':
                Z_pos += 2 * mag_params[0]/100 - z_gap
                continue
            elif mag == 'M2': 
                Ymgap = 0.05; yoke_type = 'Mag2'; mag_params[-1] = 3.20E+06; B_goal = None
        p = get_magnet_params(mag_params, Ymgap=Ymgap, z_gap=z_gap, B_goal = B_goal, yoke_type=yoke_type, resol = resol)
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)'] + z_gap
        if mag == 'M2': Z_pos += z_gap
    all_params.to_csv('magnet_params.csv')




def get_symmetry(points:np.array, B:np.array, reorder:bool = True):
   '''Applies symmetry to the computed magnetic field.'''
   points_1 = points
   points_2 = np.array([-points[:,0], points[:,1], points[:,2]]).T
   points_3 = np.array([-points[:,0], -points[:,1], points[:,2]]).T
   points_4 = np.array([points[:,0], -points[:,1], points[:,2]]).T
   B_1 = B
   B_2 = np.array([-B[:,0], B[:,1], B[:,2]]).T
   B_3 = B
   B_4 = np.array([-B[:,0], B[:,1], B[:,2]]).T
   points = np.vstack((points_1, points_2, points_3, points_4))
   B = np.vstack((B_1, B_2, B_3, B_4))
   if reorder:
      sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
      points = points[sorted_indices]
      B = B[sorted_indices]
   return points, B

def construct_grid(limits = ((0., 0., -5.),(2.5, 3.5, 5.)), 
                   resol = RESOL_DEF,
                   eps:float = 1e-12):
    '''Constructs a grid based on the limits and resolution given.'''
    (min_x, min_y, min_z), (max_x, max_y, max_z) = limits
    r_x, r_y, r_z = resol
    X, Y, Z = np.meshgrid(np.arange(min_x, max_x + r_x, r_x),
                            np.arange(min_y, max_y + r_y, r_y),
                            np.arange(min_z, max_z + r_z, r_z))
    # to avoid evaluating at 0
    #X[X == 0.0] = eps
    #Y[Y == 0.0] = eps
    #Z[Z == min_z] = min_z #+ eps
    #Z[Z == max_z] = max_z #- eps
    return X, Y, Z

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

def get_vector_field(magn_params,materials_dir):
    if magn_params['yoke_type'][0] == 'Mag1':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_1(magn_params, 0, materials_directory=materials_dir)
    elif magn_params['yoke_type'][0] == 'Mag2':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_2(magn_params, 0, materials_directory=materials_dir)
    elif magn_params['yoke_type'][0] == 'Mag3':
        points, B, M_i, M_c, Q, J = snoopy.get_vector_field_mag_3(magn_params, 0, materials_directory=materials_dir)
    else: raise ValueError(f'Invalid yoke type - Received yoke_type {magn_params["yoke_type"][0]}')
    return points.round(4).astype(np.float16), B.round(4).astype(np.float16), M_i, M_c, Q, J

def run_fem(magn_params:dict,
            materials_dir = None):
    """Runs the finite element method to compute the magnetic field.
    Parameters:
    magn_params (dict): Dictionary containing the magnets parameters.
    materials_dir (str, optional): Directory containing the materials. Defaults is None, returning the data dir in tha parent dir.
    Returns:
    dict: A dictionary containing the position points and the computed magnetic field 'B'.
    """
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/materials')
    start = time()
    points, B, M_i, M_c, Q, J = get_vector_field(magn_params, materials_dir)
    C_i, C_c, C_edf = snoopy.compute_prices(magn_params, 0, M_i, M_c, Q,materials_directory = materials_dir)
    cost = C_i + C_c + C_edf
    end = time()
    print('FEM Computation time = {} sec'.format(end - start))
    return {'points':points, 'B':B}

def simulate_and_grid(params, points):
    return get_grid_data(**run_fem(params), new_points=points)[1]

def run(magn_params:dict,
        resol = RESOL_DEF,
        d_space = ((4., 4., (-1, 30.))),
        plot_results:bool = False,
        save_results:bool = False,
        output_file:str = './outputs',
        apply_symmetry:bool = False,
        cores:int = 1,
        ):
    """Simulates the magnetic field based on given parameters and performs various operations such as applying symmetry,
    plotting results, and saving results.
    Parameters:
    magn_params (dict): Dictionary containing magnetic parameters.
    output_file (str, optional): Directory to save outputs. Defaults to './outputs'.
    apply_symmetry (bool, optional): Whether to apply symmetry to the computed magnetic field. Defaults to False.
    plot_results (bool, optional): Whether to plot the results. Defaults to False.
    save_results (bool, optional): Whether to save the results to a file. Defaults to False.
    resol (tuple, optional): Resolution of the grid. Defaults to (0.05, 0.05, 0.05).
    d_space (tuple, optional): Dimensions of the space returned. Since the problem is symmetric, it must be in the form (dx,dy,(-z_i,z_f)).
    Defaults to ((3.5, 4.5, (-15., 15.))).
    Returns:
    dict: A dictionary containing the computed points and magnetic field 'B'.
    """
    
    n_magnets = len(magn_params['yoke_type'])
    print('Starting simulation for {} magnets'.format(n_magnets))
    limits_quadrant = ((0., 0., d_space[2][0]), (d_space[0],d_space[1], d_space[2][1]))
    points = construct_grid(limits=limits_quadrant, resol=resol)

    with mp.Pool(cores) as pool:
        B = pool.starmap(simulate_and_grid, [({k: [v[i]] for k, v in magn_params.items()}, points) for i in range(0,n_magnets)])
    #B, cost = zip(*results)
    B = np.sum(B, axis=0)
    #cost = np.sum(cost)


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
              z_gap = 0.1,
              resol = RESOL_DEF,
              d_space = ((4., 4., (-1, 30.))), 
              NI_from_B_goal:bool = True,
              file_name = 'data/outputs/fields.pkl',
              cores = 1):
    '''Simulates the magnetic field for the given parameters. If save_fields is True, the fields are saved to data/outputs/fields.pkl'''
    t1 = time()
    all_params = pd.DataFrame()
    Z_pos = 0.
    for i, (mag,idx) in enumerate(new_parametrization.items()):
        mag_params = params[idx]
        if mag_params[0]<1: continue
        if mag_params[1]<1: 
            Z_pos += 2 * mag_params[0]/100 - z_gap
            continue
        if mag == 'HA': Ymgap=0.; yoke_type = 'Mag1'; B_goal = 1.9 if NI_from_B_goal else None
        elif mag in ['M1', 'M2', 'M3']: Ymgap = 0.; B_goal = 1.9 if NI_from_B_goal else None; yoke_type = 'Mag1'
        else: Ymgap = 0.; B_goal = 1.9 if NI_from_B_goal else None; yoke_type = 'Mag3'
        if fSC_mag:
            if mag == 'M1': continue
            elif mag == 'M3':
                Z_pos += 2 * mag_params[0]/100 - z_gap
                continue
            elif mag == 'M2': 
                Ymgap = 0.05; yoke_type = 'Mag2'; mag_params[-1] = 3.20E+06; B_goal = None
        p = get_magnet_params(mag_params, Ymgap=Ymgap, z_gap=z_gap, B_goal = B_goal, yoke_type=yoke_type, resol = resol)
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)'] + z_gap
        if mag == 'M2': Z_pos += z_gap
    try: all_params.to_csv(os.path.join(os.environ.get('PROJECTS_DIR', '../'), 'MuonsAndMatter/data/magnet_params.csv'), index=False)
    except: pass
    all_params = all_params.to_dict(orient='list')
    fields = run(all_params, d_space=d_space, resol=resol, apply_symmetry=False, cores=cores)
    fields['points'][:,2] += Z_init/100
    print('Magnetic field simulation took', time()-t1, 'seconds')
    if file_name is not None:
        #with open(file_name, 'wb') as f:
        #    pickle.dump(fields, f)
        np.save(file_name, fields['B'].astype(np.float16))
        print('Fields saved to', file_name)
        file_name = file_name.replace('fields', 'points')
        np.save(file_name, fields['points'].astype(np.float16))
    return fields
   


if __name__ == '__main__':
   
   MAIN_DIR = '/home/hep/lprate/projects/roxie_ship'
   parameters_filename = "/home/hep/lprate/projects/MuonsAndMatter/data/magnet_params.csv"
   output_file = "/home/hep/lprate/projects/MuonsAndMatter/data/outputs/fields_mm.npy"

   import pandas as pd
   magn_params = pd.read_csv(parameters_filename).to_dict(orient='list')
   print(magn_params)
   t1 = time()
   d = run(magn_params,resol = RESOL_DEF,output_file = output_file, save_results=True, cores = 7)
   print('total_time: ', time() - t1, ' sec')
   points = d['points']
   print('limits: ', points.min(axis=0), points.max(axis=0))
   print('shape: ', points.shape)