
import numpy as np
import os
import torch
import sys
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization/src'))
from problems import ShipMuonShieldCluster
from time import time

def extract_number_from_string(s):
    number_str = ''
    for char in s:
        if char.isdigit(): 
            number_str += char
    return int(number_str)

def concatenate_files(dir, file_number):
    all_data = []
    for file in os.listdir(dir):
        if f'_{file_number}_' not in file: continue
        with gzip.open(os.path.join(dir,file),'rb') as f:
            all_data.append(pickle.load(f))
        os.remove(os.path.join(dir,file))
    all_data = np.concatenate(all_data,axis=1)
    with gzip.open(os.path.join(dir,f'muons_data_{file_number}.pkl'),'wb') as f:
        pickle.dump(all_data,f)

def test(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 512,
        seed = 1,
        tag = ''):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.198.159',
                 port=444,
                 seed = seed)
    time1 = time()
    SHIP(phi).item()
    print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
    print('saved new file')
    print('TIME:', time.time()-t1)

def get_files(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 384,
        seed = 1):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.198.159',
                 port=444,
                 seed = seed)

    for name in os.listdir(inputs_dir):
        n_name = extract_number_from_string(name)
        print('FILE:', name)
        t1 = time()
        with gzip.open(os.path.join(inputs_dir,name), 'rb') as f:
            factor = pickle.load(f)[:,-1]
        SHIP.n_samples = factor.shape[0]
        time1 = time()
        SHIP(phi,file = n_name).item()
        print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
        concatenate_files(outputs_dir, n_name)
        print('saved new file')
        print('TIME:', time()-t1)

def get_total_hits(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 512,
        seed = 1,
        tag = ''):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.198.159',
                 port=444,
                 seed = seed)
    n_muons_total = 0
    n_muons_unweighted = 0
    n_hits_total = 0
    all_results = {}
    for name in os.listdir(inputs_dir):

        n_name = extract_number_from_string(name)
        print('FILE:', name)
        t1 = time()
        with gzip.open(os.path.join(inputs_dir,name), 'rb') as f:
            factor = pickle.load(f)[:,-1]
        SHIP.n_samples = factor.shape[0]
        n_muons = factor.sum()
        n_muons_total += n_muons
        time1 = time()
        n_hits = SHIP(phi,file = n_name).item()
        print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
        concatenate_files(outputs_dir, n_name)
        n_hits_total += n_hits
        n_muons_unweighted += len(factor)
        all_results[n_name] = (n_muons,n_hits)
        with gzip.open(os.path.join(outputs_dir,f'num_muons_hits_{tag}.pkl'), "wb") as f:
            pickle.dump(all_results, f)
        print('TIME:', time.time()-t1)
        print('N EVENTS: ', len(factor))
        print('N MUONS: ', n_muons)
        print('N_HITS: ', n_hits)
        print('Survival rate: ', n_hits/n_muons)
    return n_muons_total,n_hits_total, n_muons_unweighted

def get_loss(phi,inputs_dir:str,
        outputs_dir:str, 
        cores:int = 512,
        seed = 1,
        tag = ''):
    SHIP = ShipMuonShieldCluster(cores = cores,
                 loss_with_weight = False,
                 manager_ip='34.65.198.159',
                 port=444,
                 seed = seed)
    total_loss = 0
    all_results = {}
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
    return total_loss



sc_v6 = [0,353.078,125.083,184.834,150.193,186.812,72,51,29,46,10,7,45.6888,
         45.6888,22.1839,22.1839,27.0063,16.2448,10,31,35,31,51,11,24.7961,48.7639,8,104.732,15.7991,16.7793,3,100,192,192,2,
         4.8004,3,100,8,172.729,46.8285,2]
optimal_oliver =  [208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0,
                         38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 
                         24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0]

if __name__ == '__main__':
    INPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/full_sample'
    OUTPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs'
    import argparse
    import gzip
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument("-tag", type=str, default='')
    parser.add_argument("-n_tasks", type=int, default=512)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-inputs_dir", type=str, default=INPUTS_DIR)
    parser.add_argument("-outputs_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument("-params", type=str, default=optimal_oliver)
    #parser.add_argument("--z", type=float, default=0.1)
    #parser.add_argument("--sens_plane", type=float, default=57)
    parser.add_argument("-calc_loss", action = 'store_true')
    parser.add_argument("-only_files", action = 'store_true')
    args = parser.parse_args()
    
    if isinstance(args.params,str):
        print('Using parameters from ', args.params)
        with open(args.params, 'r') as file:
            params = []
            for line in file:
                number = float(line.strip())
                params.append(number)
    else: params = args.params
    params = torch.tensor(params)
    
    #input_dist = args.z
    #sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}

    t1 = time()
    if args.calc_loss:
        loss = get_loss(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,
        seed = args.seed,
        tag = args.tag)
        print(f'TOTAL MUONS LOSS: {loss}')
    elif args.only_files:
        t2 = time()
        get_files(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,#sensitive_film_params = sensitive_film_params,
        seed = args.seed, #input_dist=args.z,
        tag = args.tag)
    else:
        n_muons,n_hits, n_un = get_total_hits(params,args.inputs_dir,args.outputs_dir, 
        cores = args.n_tasks,#sensitive_film_params = sensitive_film_params,
        seed = args.seed, #input_dist=args.z,
        tag = args.tag)
        print(f'number of events: {n_un}')
        print(f'INPUT MUONS: {n_muons}')
        print(f'HITS: {n_hits}')
        print(f'Muons survival rate: {n_hits/n_muons}')
    t2 = time()
    
    print(f'TOTAL TIME: {(t2-t1):.3f}')
    