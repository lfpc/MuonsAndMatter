"""SHiP NC magnet get map template.
   ==========

   Compute the field map for the NC magnets.
"""
import os
import numpy as np
import pyvista as pv
import time
import gmsh
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import gzip, pickle
import roxie_evaluator
import multiprocessing as mp



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
                   resol = (0.05, 0.05, 0.05),
                   eps:float = 1e-12):
    '''Constructs a grid based on the limits and resolution given.'''
    (min_x, min_y, min_z), (max_x, max_y, max_z) = limits
    r_x, r_y, r_z = resol
    X, Y, Z = np.meshgrid(np.arange(min_x, max_x + r_x, r_x),
                            np.arange(min_y, max_y + r_y, r_y),
                            np.arange(min_z, max_z + r_z, r_z))
    # to avoid evaluating at 0
    X[X == 0.0] = eps
    Y[Y == 0.0] = eps
    Z[Z == min_z] = min_z #+ eps
    Z[Z == max_z] = max_z #- eps
    return X, Y, Z

def get_grid_data(points: np.array, B: np.array, new_points: tuple):
    '''Interpolates the magnetic field data to a new grid.'''
    t1 = time.time()
    '''Bx_out = griddata(points, B[:, 0], new_points, method='nearest', fill_value=0.0).ravel()
    By_out = griddata(points, B[:, 1], new_points, method='nearest', fill_value=0.0).ravel()
    Bz_out = griddata(points, B[:, 2], new_points, method='nearest', fill_value=0.0).ravel()'''
    new_points = np.column_stack((new_points[0].ravel(), new_points[1].ravel(), new_points[2].ravel()))
    Bx_out, By_out, Bz_out = np.zeros_like(new_points).T
    hull =  (new_points[:, 0] >= points[:, 0].min()) & (new_points[:, 0] <= points[:, 0].max()) & \
            (new_points[:, 1] >= points[:, 1].min()) & (new_points[:, 1] <= points[:, 1].max()) & \
            (new_points[:, 2] >= points[:, 2].min()) & (new_points[:, 2] <= points[:, 2].max())
    tree = cKDTree(points)
    _, idx = tree.query(new_points[hull], k=1)
    Bx_out[hull] = B[idx, 0]
    By_out[hull] = B[idx, 1]
    Bz_out[hull] = B[idx, 2]
    new_B = np.column_stack((Bx_out, By_out, Bz_out))
    print('Griddind / Interpolation time = {} sec'.format(time.time() - t1))
    return new_points, new_B

def run_fem(magn_params:dict,
            delta_air = (1.0,1.0,1.0),
            materials_dir = None):
    """Runs the finite element method to compute the magnetic field.
    Parameters:
    magn_params (dict): Dictionary containing the magnets parameters.
    delta_air (tuple, optional): Dimensions of block of air outside the magnets to simulate. Defaults to (1.0,1.0,1.0).
    materials_dir (str, optional): Directory containing the materials. Defaults is None, returning the data dir in tha parent dir.
    Returns:
    dict: A dictionary containing the position points and the computed magnetic field 'B'.
    """
    if materials_dir is None:
        materials_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

    mu_air = roxie_evaluator.ConstantPermeability(4*np.pi*1e-7)
    mu_iron = roxie_evaluator.Permeability(os.path.join(materials_dir, magn_params['material'] + '.csv'))

    lc = 2e-1 # mesh size parameter #2e-1
    element_order = 1# element oder
    max_iter = 50 # maximum number of newton iterations
    tolerance = 1e-3 # the tolerance for the cg solver
    gmsh.clear()
    if not gmsh.isInitialized():
        print('Initializing gmsh')
        gmsh.initialize()
        gmsh.model.add("make_SHiP_magnet_mesh")

    delta_x, delta_y, delta_z = delta_air

    if magn_params['yoke_type'] == 'Mag2':
        NI = magn_params['NI(A)']
        valid = True #?
        mu = [mu_iron, mu_iron, mu_air, mu_air]
        make_core_fn = roxie_evaluator.make_SHIP_sc_iron_core

    elif magn_params['yoke_type'] in ['Mag1','Mag3']: 
        NI, valid = roxie_evaluator.get_NI(magn_params['B_goal(T)'],
                                        magn_params['Xmgap1(m)'],
                                        magn_params['Xcore1(m)'],
                                        magn_params['Xvoid1(m)'],
                                        magn_params['Xyoke1(m)'],
                                        magn_params['Xmgap2(m)'],
                                        magn_params['Xcore2(m)'],
                                        magn_params['Xvoid2(m)'],
                                        magn_params['Xyoke2(m)'],
                                        magn_params['Ycore1(m)'],
                                        magn_params['Yvoid1(m)'],
                                        magn_params['Yyoke1(m)'],
                                        magn_params['Ycore2(m)'],
                                        magn_params['Yvoid2(m)'],
                                        magn_params['Yyoke2(m)'],
                                        magn_params['Z_len(m)'],
                                        magn_params['yoke_type'],
                                        mu_iron)
        mu = [mu_iron, mu_air]
        make_core_fn = roxie_evaluator.make_SHIP_iron_core
                                
    print('NI = {}'.format(NI))

    if not valid:
      print('\n')
      print('**************************************************************')
      print('WARNING! The core and return yoke dimensions are inconsistent!')
      print('**************************************************************')
      print('\n')

   
   # make the mesh

    gmsh_model, term_1, term_2 = make_core_fn( magn_params['Xmgap1(m)'],
                                                magn_params['Xcore1(m)'],
                                                magn_params['Xvoid1(m)'],
                                                magn_params['Xyoke1(m)'],
                                                magn_params['Xmgap2(m)'],
                                                magn_params['Xcore2(m)'],
                                                magn_params['Xvoid2(m)'],
                                                magn_params['Xyoke2(m)'],
                                                magn_params['Ycore1(m)'],
                                                magn_params['Yvoid1(m)'],
                                                magn_params['Yyoke1(m)'],
                                                magn_params['Ycore2(m)'],
                                                magn_params['Yvoid2(m)'],
                                                magn_params['Yyoke2(m)'],
                                                magn_params['Z_len(m)'],
                                                delta_x,
                                                delta_y,
                                                delta_z,
                                                Z_pos=magn_params['Z_pos(m)'],
                                                element_order=element_order,
                                                lc=lc,
                                                show=False)


    # Make the solver
    solver = roxie_evaluator.PoissonSolver(gmsh_model.mesh, mu, (term_1, term_2),
                                            max_iter=max_iter, quad_order=element_order+1)

    # solve the problem
    start = time.time()
    x = solver.solve(NI, tolerance=tolerance, use_cg=True) #difference hear
    # compute the field map
    points, B = solver.compute_field(x, 'B', quad_order=5)
    end = time.time()
    print('FEM Computation time = {} sec'.format(end - start))
    # points_H, H = solver.compute_field(x, 'H', quad_order=element_order+1)
    return {'points':points, 'B':B}

def simulate_and_grid(params, points):
    delta_z = 2.0 if params['B_goal(T)'] > 2 else 1.0
    return get_grid_data(**run_fem(params, delta_air = (1.0,1.0,delta_z)), new_points=points)[1]

def run(magn_params:dict,
        resol = (0.05, 0.05, 0.05),
        d_space = ((2.5, 2.5, (-0.,4.))),
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
    
    n_magnets = len(magn_params['B_goal(T)'])
    print('Starting simulation for {} magnets'.format(n_magnets))
    limits_quadrant = ((0., 0., d_space[2][0]), (d_space[0],d_space[1], d_space[2][1]))
    points = construct_grid(limits=limits_quadrant, resol=resol)
    with mp.Pool(cores) as pool:
        B = pool.starmap(simulate_and_grid,[({k:v[i] for k,v in magn_params.items()},points) for i in range(n_magnets)])
    B = np.sum(B, axis=0)
    
    '''B = 0
    for i in range(n_magnets):
        print('Simulating magnet {}'.format(i+1))
        params = {k:v[i] for k,v in magn_params.items()}
        delta_z = 2.0 if params['B_goal(T)'] > 2 else 1.0
        B += get_grid_data(**run_fem(params, delta_air = (1.0,1.0,delta_z)), new_points=points)[1]'''
    points = np.column_stack([points[i].ravel() for i in range(3)])
    if apply_symmetry:
        points,B = get_symmetry(points, B, reorder = True)
    if plot_results:
        pv.start_xvfb()
        pl = pv.Plotter()
        roxie_evaluator.plot_vector_field(pl, points[:, [2, 0, 1]] , B[:, [2, 0, 1]] , title='B in T', mag=0.1)
        #pl.add_mesh(points[:, [2, 0, 1]] , point_size=1.0, render_points_as_spheres=True, color='red')
        pl.show_grid(ztitle='Y [m]', xtitle='Z [m]', ytitle='X [m]')
        pl.add_axes(zlabel='Y [m]', xlabel='Z [m]', ylabel='X [m]')
        pl.view_isometric() 
        pl.save_graphic(os.path.join(output_file,'plot_nc.pdf'))
        print('Plot saved to', os.path.join(output_file,'plot_nc.pdf'))

    if save_results:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump({'points':points, 'B':B}, f)
        print('Results saved to', output_file)
    return {'points':points, 'B':B}
    
   
   


if __name__ == '__main__':
   
   MAIN_DIR = '/home/hep/lprate/projects/roxie_ship'
   parameters_filename = os.path.join(MAIN_DIR,'inputs', 'parameters.csv')
   output_file = os.path.join(MAIN_DIR,'outputs')

   import pandas as pd
   magn_params = pd.read_csv(parameters_filename).to_dict(orient='list')
   print(magn_params)
   t1 = time.time()
   d = run(magn_params,output_file = output_file, apply_symmetry=False, plot_results=False, save_results=True)
   print('total_time: ', time.time() - t1, ' sec')
   points = d['points']
   print('limits: ', points.min(axis=0), points.max(axis=0))
   print('shape: ', points.shape)