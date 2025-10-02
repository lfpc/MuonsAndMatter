import numpy as np
import os
import torch
import sys
PROJECTS_DIR = os.getenv('PROJECTS_DIR')
sys.path.insert(1, os.path.join(PROJECTS_DIR,'BlackBoxOptimization/src'))
from problems import ShipMuonShieldCluster
from time import time
import h5py


def _process_and_append_chunk_outputs(temp_dir: str, output_file_handle, feature_order: list):
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith('outputs_') and f.endswith('.npy')]
    if not temp_files:
        print("No temporary simulation outputs found to process for this chunk.")
        return

    all_chunk_arrays = []
    print(f"  -> Found {len(temp_files)} temporary .npy files to process.")
    for filename in temp_files:
        f_path = os.path.join(temp_dir, filename)
        data = np.load(f_path)
        if data.size > 0: all_chunk_arrays.append(data)
        os.remove(f_path)

    if not all_chunk_arrays:
        return
    concatenated_array = np.concatenate(all_chunk_arrays, axis=1).T
    print("Found data with shape:", concatenated_array.shape)
    print('Number of muons that end BEFORE of sens plane:', np.sum(concatenated_array[:,5]<(CONFIG['sensitive_plane'][0]['position']-CONFIG['sensitive_plane'][0]['dz'])))
    print('Number of muons that end AFTER of sens plane:', np.sum(concatenated_array[:,5]>(CONFIG['sensitive_plane'][0]['position']+CONFIG['sensitive_plane'][0]['dz'])))

    for j, key in enumerate(feature_order):
        column_data = concatenated_array[:, j]

        if key not in output_file_handle:
            maxshape = (None,)
            output_file_handle.create_dataset(key, data=column_data, maxshape=maxshape, chunks=True)
        else:
            dset = output_file_handle[key]
            old_size = dset.shape[0]
            dset.resize(old_size + column_data.shape[0], axis=0)
            dset[old_size:] = column_data
            
    print(f"  -> Processed and appended data for keys: {feature_order}")

def get_total_hits(phi,
                   n_split: int = 5_000_000,
                   n_muons: int = 0,
                   inputs_file: str = 'data/muons/full_sample.h5',
                   outputs_dir: str = 'data/outputs/results',
                   config: dict = {}):
    SHIP = ShipMuonShieldCluster(dimensions_phi=phi.size(-1), initial_phi=phi, **config)
    
    feature_order = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight']
    
    if not inputs_file.endswith('.h5'):
        raise ValueError("Input file must be an HDF5 file (.h5)")

    print(f"Checking HDF5 input file '{inputs_file}'...")
    with h5py.File(inputs_file, 'r') as f:
        if not all(key in f for key in feature_order):
            raise ValueError(f"Input HDF5 file is missing required keys. Expected: {feature_order}. Got: {list(f.keys())}")
        total_events = f[feature_order[0]].shape[0]
    
    print(f"Found {total_events} total events.")
    if n_muons > 0:
        total_events = min(total_events, n_muons)
        print(f"Limiting to first {total_events} events as specified.")

    n_muons_total = 0
    n_hits_total = 0
    n_muons_unweighted = 0
    all_results = {}

    n_chunks = (total_events + n_split - 1) // n_split
    print(f"Splitting into {n_chunks} chunks of size {n_split}...\n")

    final_output_file = os.path.join(outputs_dir, 'final_concatenated_results.h5')
    
    with h5py.File(final_output_file, 'w') as out_f:
        for i in range(n_chunks):
            start_idx = i * n_split
            end_idx = min((i + 1) * n_split, total_events)
            
            with h5py.File(inputs_file, 'r') as in_f:
                weights_chunk = in_f['weight'][start_idx:end_idx]

            SHIP.n_samples = len(weights_chunk)
            n_muons_chunk = weights_chunk.sum()
            
            print(f"--- Processing Chunk {i+1}/{n_chunks} (Events {start_idx}-{end_idx}) ---")
            print(f'n_events_input: {SHIP.n_samples}')
            print(f'n_particles: {n_muons_chunk:.2f}')
            
            time1 = time()

            n_hits_chunk = SHIP.simulate(phi, idx = (start_idx, end_idx)).item()

            print(f'SIMULATION FINISHED - TOOK {time()-time1:.3f} seconds')
            
            _process_and_append_chunk_outputs(outputs_dir, out_f, feature_order)

            n_hits_total += n_hits_chunk
            n_muons_total += n_muons_chunk
            n_muons_unweighted += SHIP.n_samples
            all_results[i] = (n_muons_chunk, n_hits_chunk)

            print('N MUONS: ', n_muons_chunk)
            print('N_HITS: ', n_hits_chunk)
            print('Survival rate: ', n_hits_chunk / n_muons_chunk if n_muons_chunk > 0 else 0)
            print("-" * 50)
            SHIP.uniform_fields = True

    print("\n--- Overall Summary ---")
    print(f"Total events processed: {n_muons_unweighted}")
    print(f"Total weighted muons: {n_muons_total:.2f}")
    print(f"Total final hits: {n_hits_total}")
    print(f"Overall survival rate: {n_hits_total / n_muons_total if n_muons_total > 0 else 0:.6f}")
    print(f"Data saved to {final_output_file}")    
    return n_muons_total, n_hits_total, n_muons_unweighted


new_parametrization = ShipMuonShieldCluster.parametrization
if __name__ == '__main__':
    INPUTS_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample.h5'
    OUTPUTS_DIR = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results'
    import argparse
    import json
    # Load config file
    CONFIG_PATH = os.path.join(PROJECTS_DIR, 'cluster', 'config.json')
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.pop("data_treatment", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_split", type=int, default=5_000_000)
    parser.add_argument("-n_muons", type=int, default=0)
    parser.add_argument("-inputs_file", type=str, default=INPUTS_FILE)
    parser.add_argument("-outputs_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument("-params", type=str, default='tokanut_v5')
    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    if not os.path.exists(os.path.dirname(args.outputs_dir)):
        os.makedirs(os.path.dirname(args.outputs_dir))
    
    if args.params == 'test':
        params_input = input("Enter the params as a Python list (e.g., [1.0, 2.0, 3.0]): ")
        params = eval(params_input)
    elif args.params in ShipMuonShieldCluster.params.keys():
        params = ShipMuonShieldCluster.params[args.params]
    elif os.path.isfile(args.params):
        with open(args.params, "r") as txt_file:
            params = [float(line.strip()) for line in txt_file]
    else: 
        raise ValueError(f"Invalid params: {args.params}. Must be a valid parameter name or a file path. \
                         Avaliable names: {', '.join(ShipMuonShieldCluster.params.keys())}.")
    params = torch.as_tensor(params, dtype=torch.float32)



    t1 = time()
    n_muons, n_hits, n_un = get_total_hits(params, inputs_file=args.inputs_file, outputs_dir = args.outputs_dir, 
        n_split=args.n_split, n_muons = args.n_muons,
        config=CONFIG)
    print(f'number of events: {n_un}')
    print(f'INPUT MUONS: {n_muons}')
    print(f'HITS: {n_hits}')
    print(f'Muons survival rate: {n_hits/n_muons}')
    t2 = time()
    
    print(f'TOTAL TIME: {(t2-t1):.3f}')
