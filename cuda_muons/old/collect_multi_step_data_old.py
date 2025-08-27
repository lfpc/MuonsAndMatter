import argparse
import h5py
import time

import numpy as np
import json
import multiprocessing as mp
import os
from muon_slabs import simulate_muon, initialize, collect, set_field_value, set_kill_momenta, kill_secondary_tracks, is_single_step, collect_from_sensitive
import lib.gigantic_sphere as sphere_design
import functools


# Function to simulate a batch of muons in a single
def simulate_muon_batch(num_simulations, detector, step_goal=0.05, single_step=False, initial_momenta_bounds=(10, 400)):
    np.random.seed((os.getpid() * int(time.time())) % 2**32)
    batch_data = {
        'initial_momenta': [],
        'px': [],
        'py': [],
        'pz': [],
        'step_length': [],
    }
    initialize(np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), json.dumps(detector))
    kill_secondary_tracks(True)
    i = 0
    while i < num_simulations:
        initial_momenta = np.random.uniform(*initial_momenta_bounds)
        batch_data['initial_momenta'].append(initial_momenta)
        simulate_muon(0, 0, initial_momenta, 1, 0, 0, 0)
        data = collect_from_sensitive()
        index = 0#np.searchsorted(np.cumsum(data['step_length']), n_steps*0.02)
        if len(data['z'])>0 and 13 in np.abs(data['pdg_id']): 
            while int(abs(data['pdg_id'][index])) != 13:
                index += 1
        else: continue
        batch_data['px'].append(data['px'][index])
        batch_data['py'].append(data['py'][index])
        batch_data['pz'].append(data['pz'][index])
        i += 1
    assert len(batch_data['px']) == num_simulations, f"Expected {num_simulations} simulations, got {len(batch_data['px'])}"
    return batch_data

def parallel_simulations(num_sims, detector, num_processes=4, step_goal=0.05, output_file='muons_data.h5', single_step=False, initial_momenta_bounds=(10, 400)):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('initial_momenta', (0,), maxshape=(None,), dtype='f8')
        f.create_dataset('px', (0,), maxshape=(None,), dtype='f8')
        f.create_dataset('py', (0,), maxshape=(None,), dtype='f8')
        f.create_dataset('pz', (0,), maxshape=(None,), dtype='f8')
    simulate_fn = functools.partial(simulate_muon_batch, detector=detector, step_goal=step_goal, single_step=single_step, initial_momenta_bounds=initial_momenta_bounds)
    chunk_size = num_sims // num_processes
    remainder = num_sims % num_processes
    chunks = [chunk_size + (1 if i < remainder else 0) for i in range(num_processes)]
    total_written = 0
    assert len(chunks) == num_processes, f"Expected {num_processes} chunks, got {len(chunks)}"
    with h5py.File(output_file, 'a') as f:
        with mp.Pool(processes=num_processes) as pool:
            for batch_result in pool.imap_unordered(simulate_fn, chunks):
                current_size = f['pz'].shape[0]
                batch_size = len(batch_result['pz'])
                print(batch_size)
                new_size = current_size + batch_size
                f['px'].resize((new_size,))
                f['py'].resize((new_size,))
                f['pz'].resize((new_size,))
                f['initial_momenta'].resize((new_size,))
                f['initial_momenta'][current_size:new_size] = batch_result['initial_momenta']
                f['px'][current_size:new_size] = batch_result['px']
                f['py'][current_size:new_size] = batch_result['py']
                f['pz'][current_size:new_size] = batch_result['pz']

                f.flush()
                total_written += batch_size
                print(f"Written batch of {batch_size} simulations. Total: {total_written}/{num_sims}")
    print(f"Completed writing {total_written} simulations to {output_file}")
    return output_file

def main(cores=16, num_sims=1_000_000, step_goal=0.05, single_step = False, initial_momenta=(10, 400), tag:str = '5'):
    folder = f"data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    sensitive_film = {"name": "SensitiveFilm", "z_center": step_goal, "dz": 0.001, "dx": 20, "dy": 20}
    detector = sphere_design.get_design(mag_field = [0.,0.,0.], sens_film=sensitive_film)
    detector["store_primary"] = False
    detector["store_all"] = False
    #detector["limits"]["max_step_length"] = step_goal
    output_file = os.path.join(folder, f"muon_data_energy_loss_sens_step_{tag}.h5")
    parallel_simulations(num_sims, detector, num_processes=cores, step_goal=step_goal, output_file=output_file, single_step=single_step, initial_momenta_bounds=initial_momenta)
    print(f"Data saved to {output_file}")

def main_split(cores:int=16,num_sims:int = 10_000_000, step_goal:float = 0.02, initial_momenta=(10, 400), single_step:bool = False):
    n_sim_split = min(1_000_000, num_sims)
    n = 0
    i = 0
    while n<num_sims:
        main(cores, n_sim_split, step_goal, single_step=single_step, initial_momenta=initial_momenta, tag=str(int(step_goal*100)) + str(i))
        n += n_sim_split
        print(f"Finished {n} simulations")
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run muon simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=16, help='Number of CPU cores to use')
    parser.add_argument('--num_sims', type=int, default=1_000_000, help='Number of simulations to run')
    parser.add_argument('--step_goal', type=float, default=0.02, help='Step goal for simulation')
    parser.add_argument('--split', action='store_true', help='Use main_split instead of main')
    parser.add_argument('--multi_step', dest='single_step', action='store_false', help='Return full data, without transformation to single step')
    parser.add_argument('--initial_momenta', type=float, nargs=2, default=(10, 300), help='Range of initial momenta for muons (min, max)')
    args = parser.parse_args()
    if args.split:
        main_split(cores=args.cores, num_sims=args.num_sims, step_goal=args.step_goal, single_step=args.single_step, initial_momenta=args.initial_momenta)
    else:
        main(cores=args.cores, num_sims=args.num_sims, step_goal=args.step_goal, tag=int(args.step_goal*100), single_step=args.single_step, initial_momenta=args.initial_momenta)