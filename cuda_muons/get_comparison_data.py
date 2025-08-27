
from muon_slabs import simulate_muon, initialize, collect, is_single_step, kill_secondary_tracks, collect_from_sensitive
from tqdm import tqdm
import pickle
import json
import lib.gigantic_sphere as sphere_design
from faster_muons.fast_muons_main_3 import run as run_cuda_simulation
import numpy as np  
import argparse
import multiprocessing as mp



def run_geant4_sim(muons, n_steps=500, mag_field=[0., 0., 0.]):
    sensitive_film = {"name": "SensitiveFilm", "z_center": n_steps*0.02, "dz": 0.001, "dx": 20, "dy": 20}
    detector = sphere_design.get_design(mag_field=mag_field, sens_film=sensitive_film)
    detector["store_primary"] = False
    detector["store_all"] = False
    initialize(0, 4, 4, 5, json.dumps(detector))
    kill_secondary_tracks(True)
    #is_single_step(False)
    results = {
        'px': [],
        'py': [],
        'pz': [],
        'x': [],
        'y': [],
        'z': [],
        'step_length': [],
    }
    for muon in muons:
        simulate_muon(*muon[:3], 1, *muon[3:6])
        data = collect_from_sensitive()
        index = 0
        if len(data['z'])>0 and 13 in np.abs(data['pdg_id']): 
            while int(abs(data['pdg_id'][index])) != 13:
                index += 1
        else: continue

        results['px'].append(data['px'][index])
        results['py'].append(data['py'][index])
        results['pz'].append(data['pz'][index])
        results['x'].append(data['x'][index])
        results['y'].append(data['y'][index])
        results['z'].append(data['z'][index])
        #results['step_length'].append(data['step_length'][index])
    return results

def main(muons, n_steps=500, mag_field=[0., 0., 0.]):
    file_cuda = 'data/outputs_cuda.pkl'
    file_geant4 = 'data/outputs_geant4.pkl'
    n_cores = 90
    outputs_cuda = run_cuda_simulation(muons, mag_field = mag_field, histogram_file='data/alias_histograms.pkl', save_dir = None, n_steps=n_steps)
    print("CUDA simulation completed.")
    muons_split = np.array_split(muons, n_cores)
    with mp.Pool(n_cores) as pool:
        geant4_results = pool.starmap(run_geant4_sim, [(muon_batch, n_steps, mag_field) for muon_batch in muons_split])
    print("Geant4 simulation completed.")
    outputs_geant4 = {}
    for key in geant4_results[0].keys():
        outputs_geant4[key] = np.concatenate([np.array(res[key]) for res in geant4_results], axis=0)
    print("Data collection completed.")
    with open(file_cuda, 'wb') as f:
        pickle.dump(outputs_cuda, f)
    print("CUDA data saved to ", file_cuda)
    with open(file_geant4, 'wb') as f:
        pickle.dump(outputs_geant4, f)
    print("Geant4 data saved to ", file_geant4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run muon simulations with specified number of muons.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_muons', type=int, default=int(5e5), help='Number of muons to simulate')
    parser.add_argument('--n_steps', type=int, default=500, help='Number of steps to simulate')
    parser.add_argument('--mag_field', type=float, nargs=3, default=[0., 0., 0.], help='Magnetic field vector components (Bx, By, Bz)')
    args = parser.parse_args()
    n_muons = args.n_muons
    initial_momenta = np.array([[0.,0.,197.5]])
    initial_positions = np.array([[0.,0.,0.]])
    muons = np.concatenate((initial_momenta, initial_positions), axis=1)*np.ones((n_muons,1))
    main(muons, n_steps=args.n_steps, mag_field=list(args.mag_field))
