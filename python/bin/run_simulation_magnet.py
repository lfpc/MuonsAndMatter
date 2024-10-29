
import numpy as np
#from lib.ship_muon_shield import get_design_from_params
#from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
#from plot_magnet import plot_magnet, construct_and_plot
import csv

def get_roxie_params(params, 
                     SC_mag:bool = True):
    params = np.round(np.asarray(params) / 100, 2) #convert to meters
    if len(params) == 42:
        idx = np.array([1,12,13,14,15,16,17])
        params = params[idx] #params for superconducting magnet
    Ymgap = 0.05
    Xmgap = 0
    ratio_yoke = 3 if SC_mag else 1
    d = {'yoke_type': ['Mag2'],
        'coil_type': ['Racetrack'],
        'disc_x': [10],
        'disc_z': [10],
        'bias': [1.5],
        'mu_r': [1],
        'J(A/mm2)': [50],
        'N1': [3],
        'N2': [10],
        'B_goal(T)': [5.1],
        'delta_x(mm)': [1],
        'delta_y(mm)': [1],
        'Z_pos(m)': [0.02],
        'Xmgap1(m)': [Xmgap],
        'Xmgap2(m)': [Xmgap],

        'Z_len(m)': [2 * params[0]],
        'Xcore1(m)': [params[1] + Xmgap],
        'Xvoid1(m)': [params[1] + params[5] + Xmgap],
        'Xyoke1(m)': [params[1] + params[5] + ratio_yoke * params[1] + Xmgap],
        'Xcore2(m)': [params[2] + Xmgap],
        'Xvoid2(m)': [params[2] + params[6] + Xmgap],
        'Xyoke2(m)': [params[2] + params[6] + ratio_yoke * params[2] + Xmgap],
        'Ycore1(m)': [params[3]],
        'Yvoid1(m)': [params[3] + Ymgap],
        'Yyoke1(m)': [params[3] + ratio_yoke * params[1] + Ymgap], #0.33 na tabela deles???
        'Ycore2(m)': [params[4]],
        'Yvoid2(m)': [params[4] + Ymgap],
        'Yyoke2(m)': [params[4] + ratio_yoke * params[2] + Ymgap]}
    #with open("/home/hep/lprate/projects/roxie_ship/inputs/parameters.csv", "w", newline="") as f:
    #    w = csv.DictWriter(f, d.keys())
    #    w.writeheader()
    #    w.writerow(d)
    return d

from scipy.interpolate import RegularGridInterpolator, griddata
def interpolate_B(old_points,new_points,B, method):
    Bx, By, Bz = B  # Split B into three components
    #x, y, z = old_points  # Unpack the original grid points
    print(old_points.shape)
    print(Bx.shape)
    print(new_points.shape)

    Bx_new = griddata(old_points, Bx, new_points, method=method)
    By_new = griddata(old_points, By, new_points, method=method)
    Bz_new = griddata(old_points, Bz, new_points, method=method)

    return np.stack((Bx_new, By_new, Bz_new), axis=0)  # Shape will be (3, Mx, My, Mz)




def interpolate_B(old_points,new_points,B, method):
    x,y,z = old_points.T
    x = np.sort(np.unique(x))
    y = np.sort(np.unique(y))
    z = np.sort(np.unique(z))
    Nx, Ny, Nz = len(np.unique(x)), len(np.unique(y)), len(np.unique(z))
    Mx, My, Mz = len(np.unique(new_points[:,0])), len(np.unique(new_points[:,1])), len(np.unique(new_points[:,2]))

    # Separate B into its components along each axis
    Bx, By, Bz = B.T

    # Interpolators for each component
    print(x)
    Bx_interpolator = RegularGridInterpolator((x, y, z), Bx.reshape(Nx, Ny, Nz))
    By_interpolator = RegularGridInterpolator((x, y, z), By.reshape(Nx, Ny, Nz))
    Bz_interpolator = RegularGridInterpolator((x, y, z), Bz.reshape(Nx, Ny, Nz))

    # Interpolate each component of B at the new points
    Bx_new = Bx_interpolator(new_points).reshape(Mx, My, Mz)
    By_new = By_interpolator(new_points).reshape(Mx, My, Mz)
    Bz_new = Bz_interpolator(new_points).reshape(Mx, My, Mz)

    # Combine interpolated components back into a single array
    return np.array([Bx_new, By_new, Bz_new])

if __name__ == '__main__':
    import sys
    import pathlib
    import subprocess
    sys.path.append("/home/hep/lprate/projects/roxie_ship/scripts")
    sys.path.append("/home/hep/lprate/projects/roxie_ship/roxie_evaluator/scripts")
    from generate_datafiles import main as generate_datafiles
    #from lib.reference_designs.params import sc_v6
    sc_v6 = [0,353.078,125.083,184.834,150.193,186.812,72,51,29,46,10,7,45.6888,
         45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
         4.8004,3,100,8,172.729,46.8285,2]

    DIR = pathlib.Path('/home/hep/lprate/projects/roxie_ship')
    output_dir = DIR/'outputs'
    input_dir = DIR/'inputs'    
    container_dir=DIR/'containers'

    input_df = get_roxie_params(sc_v6, SC_mag=True)
    
    #generate_datafiles(input_df, input_dir, output_dir/"datafiles", container_dir)
    command = ["apptainer", "exec", "--bind",DIR.resolve(),container_dir/"roxie_evaluator.sif", "python3", "/home/hep/lprate/projects/roxie_ship/roxie_evaluator/scripts/SHiP_sc_magnet_get_map.py"]
    #try:
    #    result = subprocess.run(command, check=True, text=True, capture_output=True)
    #    print("Output:", result.stdout)
    #except subprocess.CalledProcessError as e:
    #    print("Error:", e.stderr)

    import gzip
    import pickle
    with gzip.open(output_dir/'points.pkl', 'rb') as f:
        data = pickle.load(f)
    points = data['points']
    B = data['B']

    B = np.split(B, 4, axis=0)[0]
    points = points[:B.shape[0], :]
    # Sort the x component and keep the relationship with B values
    points_new = np.array(np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), 40),
        np.linspace(points[:, 1].min(), points[:, 1].max(), 40),
        np.linspace(points[:, 2].min(), points[:, 2].max(), 70),
        indexing='ij'
    )).reshape(-1,3)

    from time import time
    all_B = []
    for method in ["nearest", "linear", "cubic"]:
        t1 = time()
        B_new = interpolate_B(points, points_new, B, method)
        print(f"Method: {method}, Time: {time() - t1}")
        print('SHAPE', B_new.shape)
        all_B.append(B_new)
    print('difference linear nearest', np.abs(all_B[1] - all_B[0]).mean())
    print('difference cubic linear', np.abs(all_B[2] - all_B[1]).mean())
    print('difference cubic nearest', np.abs(all_B[2] - all_B[0]).mean())
    