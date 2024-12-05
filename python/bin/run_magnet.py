
import numpy as np
import csv
import subprocess
import sys
import pathlib
sys.path.append("/home/hep/lprate/projects/roxie_ship/scripts")
sys.path.append("/home/hep/lprate/projects/roxie_ship/roxie_evaluator/scripts")
from os import environ
apptainer_cmd = environ.get('APPTAINER_CMD', 'apptainer') #when using this script in a container, 'apptainer' might not have the path defined
from generate_datafiles import main as generate_datafiles
from time import time
import argparse

DIR = pathlib.Path('/home/hep/lprate/projects/roxie_ship')

def get_magnet_params(params, 
                     Ymgap:float = 0.05,
                     z_gap:float = 0.1,
                     B_goal:float = 5.1,
                     save_dir:pathlib.Path = None,
                     yoke_type:str = 'Mag1'):
    #params = np.asarray(params) / 100 #convert to meters
    ratio_yoke = params[7]
    Xmgap = params[8]
    d = {'yoke_type': yoke_type,
        'coil_type': 'Racetrack',
        'material': 'bhiron_1', #check material and etc for SC
        'resol_x(m)': 0.05,
        'resol_y(m)': 0.05,
        'resol_z(m)': 0.05,
        'disc_x': 10,
        'disc_z': 10,
        'bias': 1.5,
        'mu_r': 1,
        'J(A/mm2)': 50, 
        'N1': 3,
        'N2': 10,
        'B_goal(T)': B_goal,
        'delta_x(mm)': 1,
        'delta_y(mm)': 1,
        'Z_pos(m)': -1*params[0],#[0.02],
        'Xmgap1(m)': Xmgap,
        'Xmgap2(m)': Xmgap,
        'Z_len(m)': 2 * params[0] - z_gap,
        'Xcore1(m)': params[1] + Xmgap,
        'Xvoid1(m)': params[1] + params[5] + Xmgap,
        'Xyoke1(m)': params[1] + params[5] + ratio_yoke * params[1] + Xmgap,
        'Xcore2(m)': params[2] + Xmgap,
        'Xvoid2(m)': params[2] + params[6] + Xmgap,
        'Xyoke2(m)': params[2] + params[6] + ratio_yoke * params[2] + Xmgap,
        'Ycore1(m)': params[3],
        'Yvoid1(m)': params[3] + Ymgap,
        'Yyoke1(m)': params[3] + ratio_yoke * params[1] + Ymgap, #0.33 na tabela deles???
        'Ycore2(m)': params[4],
        'Yvoid2(m)': params[4] + Ymgap,
        'Yyoke2(m)': params[4] + ratio_yoke * params[2] + Ymgap}
    if save_dir is not None:
        with open(save_dir/"parameters.csv", "w", newline="") as f:
            w = csv.DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)
    return d

def run_roxie(params, input_dir, output_dir, container_dir, generate_files=True):
    d = get_magnet_params(params, SC_mag=True)
    with open(input_dir/"parameters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)
    generate_datafiles(d, input_dir, output_dir/"datafiles", container_dir)
    command = [apptainer_cmd, "exec", "--bind",DIR.resolve(),container_dir/"roxie_evaluator.sif", "python3", DIR/"roxie_evaluator/scripts/SHiP_sc_magnet_get_map.py"]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Magnet simulation successfull; output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

def run_fem(params, input_dir, container_dir,  Ymgap=0.05, z_gap=0.1, B_goal=5.1):
    d = get_magnet_params(params, Ymgap=Ymgap, z_gap=z_gap, B_goal=B_goal)
    with open(input_dir/"parameters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)
    #command = [apptainer_cmd, "exec", "--bind",DIR.resolve(),container_dir/"roxie_evaluator.sif", "python3", DIR/"roxie_evaluator/scripts/SHiP_nc_magnet_get_map.py"]
    command = [apptainer_cmd, "exec", "--bind",DIR.resolve(), "--bind",'/home/hep/lprate/projects',"--bind","/disk/users/lprate/projects",container_dir/"roxie_evaluator.sif", "python3", "/home/hep/lprate/projects/MuonsAndMatter/python/lib/magnet_simulations.py"]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Magnet simulation successfull; output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run magnet simulation")
    parser.add_argument('--keep_files',dest = 'generate_files', action='store_false', help='Keep intermediate files')
    args = parser.parse_args()
    #from lib.reference_designs.params import sc_v6
    #sc_v6 = [0,353.078,125.083,184.834,150.193,186.812,
    #         72,51,29,46,10,7,
    #         45.6888,45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
    #     4.8004,3,100,8,172.729,46.8285,2]
    sc_v6 = np.array([3.531 , 0.457, 0.457, 0.222, 0.222, 0.270, 0.162, 1.0,0.0])

    output_dir = DIR/'outputs'
    input_dir = DIR/'inputs'    
    container_dir=DIR/'containers'
    t1 = time()
    #run_roxie(sc_v6, input_dir, output_dir, container_dir, args.generate_files)
    run_fem(sc_v6, input_dir, container_dir, B_goal = 1.7)
    print("Time:", time()-t1)


    