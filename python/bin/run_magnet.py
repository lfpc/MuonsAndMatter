
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


DIR = pathlib.Path('/home/hep/lprate/projects/roxie_ship')

def get_roxie_params(params, 
                     SC_mag:bool = True,
                     Xmgap:float = 0.0,
                     Ymgap:float = 0.05,
                     z_gap:float = 0.1,
                     B_goal:float = 5.1,
                     save_dir:pathlib.Path = None):
    params = np.asarray(params) / 100 #convert to meters
    if len(params) == 42:
        idx = np.array([1,12,13,14,15,16,17])
        params = params[idx] #params for superconducting magnet
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
        'B_goal(T)': [B_goal],
        'delta_x(mm)': [1],
        'delta_y(mm)': [1],
        'Z_pos(m)': [0.00],#[0.02],
        'Xmgap1(m)': [Xmgap],
        'Xmgap2(m)': [Xmgap],

        'Z_len(m)': [2 * params[0] - z_gap],
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
    if save_dir is not None:
        with open(save_dir/"parameters.csv", "w", newline="") as f:
            w = csv.DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)
    return d

def run_roxie(params, input_dir, output_dir, container_dir):
    d = get_roxie_params(params, SC_mag=True)
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


if __name__ == '__main__':
    #from lib.reference_designs.params import sc_v6
    sc_v6 = [0,353.078,125.083,184.834,150.193,186.812,72,51,29,46,10,7,45.6888,
         45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
         4.8004,3,100,8,172.729,46.8285,2]
    #sc_v6 = np.array([3.53078 , 0.456888, 0.456888, 0.221839, 0.221839, 0.270063,
    #   0.270063])*100

    output_dir = DIR/'outputs'
    input_dir = DIR/'inputs'    
    container_dir=DIR/'containers'
    t1 = time()
    run_roxie(sc_v6, input_dir, output_dir, container_dir)
    print("Time:", time()-t1)


    