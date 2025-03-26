
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
    for file in os.listdir(direc):
        parts = file.split('_')
        assert len(parts) == 3
        if int(parts[1]) != file_number: continue
        idx = int(parts[2][:-4])  # Remove '.npy'
        data_idx[idx] = np.load(os.path.join(direc, file))
        os.remove(os.path.join(direc,file))
    all_data = []
    for idx in sorted(data_idx.keys()):
        data = data_idx[idx]
        if data.shape != (8,):
            all_data.append(data)
    all_data = np.concatenate(all_data,axis=1)
    np.save(os.path.join(direc_output, f'muonsdata_{file_number}.npy'), all_data)

def get_files(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 384,
        seed = 1,
        field_map = False):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.139.30',
                 port=444,
                 seed = seed,
                 simulate_fields=field_map,
                 apply_det_loss = False
                 )

    for name in os.listdir(inputs_dir):
        n_name = extract_number_from_string(name)
        print('FILE:', name)
        t1 = time()
        with open(os.path.join(inputs_dir,name), 'rb') as f:
            factor = pickle.load(f)[:,-1]
        SHIP.n_samples = factor.shape[0]
        time1 = time()
        SHIP(phi,file = n_name).item()
        print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
        concatenate_files(outputs_dir, n_name)
        print('saved new file')
        print('TIME:', time()-t1)
        SHIP.simulate_fields = False

def get_total_hits(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 512,
        n_files = 67,
        seed = 1,
        tag = '',
        field_map = False,
        hybrid = False):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 apply_det_loss = False,
                 dimensions_phi = phi.size(-1),
                 manager_ip='34.65.139.30',
                 port=444,
                 seed = seed,
                 simulate_fields=field_map,
                 fSC_mag = hybrid,
                 fields_file = os.path.join(PROJECTS_DIR,'MuonsAndMatter/data/outputs/fields.npy'),
                 )
    n_muons_total = 0
    n_muons_unweighted = 0
    n_hits_total = 0
    all_results = {}
    print('LENGTH:', SHIP.get_total_length(phi))
    print('COST:', SHIP.get_total_cost(phi))
    for name in os.listdir(inputs_dir)[:n_files]:
        n_name = extract_number_from_string(name)
        if n_name == 0: continue
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
        concatenate_files('/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results', n_name, outputs_dir)
        n_hits_total += n_hits
        n_muons_unweighted += len(factor)
        all_results[n_name] = (n_muons,n_hits)
        #with open(os.path.join(outputs_dir,f'num_muons_hits_{tag}.pkl'), "wb") as f:
        #    pickle.dump(all_results, f)
        print('TIME:', time()-t1)
        print('N EVENTS: ', len(factor))
        print('N MUONS: ', n_muons)
        print('N_HITS: ', n_hits)
        print('Survival rate: ', n_hits/n_muons)
        SHIP.simulate_fields = False
    return n_muons_total,n_hits_total, n_muons_unweighted

def get_loss(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 512,
        seed = 1,
        tag = '',
        field_map = False):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.139.30',
                 port=444,
                 seed = seed,
                 simulate_fields=field_map,
                 apply_det_loss = False)
    total_loss = 0
    all_results = {}
    print('LENGTH:', SHIP.get_total_length(phi))
    print('WEIGHT:', SHIP.get_weight(phi))
    for name in os.listdir(inputs_dir):
        n_name = extract_number_from_string(name)
        print('FILE:', name)
        t1 = time.time()
        with gzip.open(os.path.join(inputs_dir,name), 'rb') as f:
            factor = pickle.load(f)[:,-1]
        SHIP.n_samples = factor.shape[0]
        loss = SHIP(phi,file = n_name).item()-1
        total_loss += loss
        all_results[n_name] = loss
        with gzip.open(os.path.join(outputs_dir,f'total_loss_{tag}.pkl'), "wb") as f:
            pickle.dump(all_results, f)
        print('TIME:', time.time()-t1)
        print('MUONS LOSS: ', loss)
        SHIP.simulate_fields = False
    return total_loss


new_parametrization = ShipMuonShieldCluster.parametrization
if __name__ == '__main__':
    INPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample'
    OUTPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-tag", type=str, default='')
    parser.add_argument("-n_tasks", type=int, default=512)
    parser.add_argument("-n_files", type=int, default=67)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-inputs_dir", type=str, default=INPUTS_DIR)
    parser.add_argument("-outputs_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument("-params", type=str, default='sc_v6')
    parser.add_argument("-warm", dest = "hybrid",action = 'store_false')
    parser.add_argument("-extra_magnet", action = 'store_true')
    parser.add_argument("-field_map", action = 'store_true')
    parser.add_argument("-calc_loss", action = 'store_true')
    parser.add_argument("-only_files", action = 'store_true')
    args = parser.parse_args()
    
    if args.params == 'sc_v6': params = ShipMuonShieldCluster.sc_v6
    elif args.params == 'oliver': params = ShipMuonShieldCluster.old_warm_opt
    elif args.params == 'warm_opt': params = ShipMuonShieldCluster.warm_opt
    elif args.params == 'melvin': params = ShipMuonShieldCluster.warm_opt_scaled
    elif args.params == 'melvin_2': params = ShipMuonShieldCluster.warm_opt_scaled_2
    elif args.params == 'warm_baseline': params = ShipMuonShieldCluster.warm_scaled_baseline
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])

    params = torch.as_tensor(params, dtype=torch.float32)

    # Create outputs directory if it doesn't exist
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    t1 = time()
    if args.calc_loss:
        loss = get_loss(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,
        seed = args.seed,
        tag = args.tag,
        field_map = args.field_map)
        print(f'TOTAL MUONS LOSS: {loss}')
    elif args.only_files:
        get_files(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,
        seed = args.seed,
        field_map = args.field_map)
    else:
        n_muons,n_hits, n_un = get_total_hits(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,
        n_files=args.n_files,
        seed = args.seed, 
        tag = args.tag,
        field_map = args.field_map,
        hybrid = args.hybrid)
        print(f'number of events: {n_un}')
        print(f'INPUT MUONS: {n_muons}')
        print(f'HITS: {n_hits}')
        print(f'Muons survival rate: {n_hits/n_muons}')
    t2 = time()
    
    print(f'TOTAL TIME: {(t2-t1):.3f}')
    