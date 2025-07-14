import numpy as np
import os
import torch
import sys
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization/src'))
from problems import ShipMuonShieldCluster
from time import time
import gzip
import pickle

def extract_number_from_string(s):
    number_str = ''
    for char in s:
        if char.isdigit(): 
            number_str += char
    return int(number_str)

def concatenate_files(direc, file_number, direc_output = None):
    if direc_output is None: direc_output = direc
    data_idx = {}
    for f in os.listdir(direc):
        parts = f.split('_')
        assert len(parts) == 3, parts
        if int(parts[1]) != file_number: continue
        idx = int(parts[2][:-4])  # Remove '.npy'
        data_idx[idx] = np.load(os.path.join(direc, f))
        os.remove(os.path.join(direc,f))
    all_data = []
    for idx in sorted(data_idx.keys()):
        data = data_idx[idx]
        if data.shape != (8,):
            all_data.append(data)
    all_data = np.concatenate(all_data,axis=1)
    np.save(os.path.join(direc_output, f'muonsdata_{file_number}.npy'), all_data)

def get_total_hits(phi,
                   inputs_dir:str,
                    outputs_dir:str, 
                    n_files:int,
                    config:dict):
    SHIP = ShipMuonShieldCluster(dimensions_phi=phi.size(-1), **config)
    n_muons_total = 0
    n_muons_unweighted = 0
    n_hits_total = 0
    all_results = {}
    print('LENGTH:', SHIP.get_total_length(phi))
    print('COST:', SHIP.get_total_cost(phi))
    for name in os.listdir(inputs_dir)[:n_files]:
        n_name = extract_number_from_string(name)
        print('FILE:', name)
        t1 = time()
        with gzip.open(os.path.join(inputs_dir,name), 'rb') as f:
            factor = pickle.load(f)[:,-1]
        SHIP.n_samples = factor.shape[0]
        n_muons = factor.sum()
        print(f'n_events_input: {SHIP.n_samples}')
        print(f'n_particles: {n_muons}')

        n_muons_total += n_muons
        time1 = time()
        n_hits = SHIP.simulate(phi,file = n_name).item()-1
        print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
        #concatenate_files('/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results', n_name, outputs_dir)
        n_hits_total += n_hits
        n_muons_unweighted += len(factor)
        all_results[n_name] = (n_muons,n_hits)
        print('TIME:', time()-t1)
        print('N EVENTS: ', len(factor))
        print('N MUONS: ', n_muons)
        print('N_HITS: ', n_hits)
        print('Survival rate: ', n_hits/n_muons)
        SHIP.simulate_fields = False
    return n_muons_total,n_hits_total, n_muons_unweighted



new_parametrization = ShipMuonShieldCluster.parametrization
if __name__ == '__main__':
    INPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample'
    OUTPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs/trajectories_full_sample/'
    import argparse
    import json
    # Load config file
    CONFIG_PATH = os.path.join(PROJECTS_DIR, 'cluster', 'config.json')
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_files", type=int, default=67)
    parser.add_argument("-inputs_dir", type=str, default=INPUTS_DIR)
    parser.add_argument("-outputs_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument("-params", type=str, default='sc_v6')
    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    # Remove all the per-parameter extraction, just pass CONFIG
    if hasattr(ShipMuonShieldCluster, args.params):
        params = getattr(ShipMuonShieldCluster, args.params)
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])

    params = torch.as_tensor(params, dtype=torch.float32)


    t1 = time()
    n_muons, n_hits, n_un = get_total_hits(params, args.inputs_dir, args.outputs_dir, 
        n_files=args.n_files,
        config=CONFIG)
    print(f'number of events: {n_un}')
    print(f'INPUT MUONS: {n_muons}')
    print(f'HITS: {n_hits}')
    print(f'Muons survival rate: {n_hits/n_muons}')
    t2 = time()
    
    print(f'TOTAL TIME: {(t2-t1):.3f}')
