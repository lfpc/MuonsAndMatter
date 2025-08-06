import argparse
import h5py
import time

import numpy as np
import json
import multiprocessing as mp
import os
from muon_slabs import add, simulate_muon, initialize, collect, set_field_value, set_kill_momenta, kill_secondary_tracks, is_single_step
import random
import lib.gigantic_sphere as sphere_design
import string
import functools

def transform_to_single_step(px, py, pz, step_length, step_goal):
    """Transform the data to a single step with the specified step_goal.
    This function modifies the last step to ensure that the total distance is exactly step_goal.
    """
    total_dists = np.cumsum(step_length)
    assert total_dists[-1] >= step_goal and total_dists[-2] < step_goal, "Total distance must be greater than or equal to step_goal"
    scale_factor = (step_goal-total_dists[-2]) / (total_dists[-1] - total_dists[-2])
    px = px[-2] + (px[-1] - px[-2]) * scale_factor
    py = py[-2] + (py[-1] - py[-2]) * scale_factor
    pz = pz[-2] + (pz[-1] - pz[-2]) * scale_factor
    step_length = step_goal
    return px, py, pz, step_length


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
    for i in range(num_simulations):
        initial_momenta = np.random.uniform(*initial_momenta_bounds)
        batch_data['initial_momenta'].append(initial_momenta)
        step = [0]
        px = [0]
        py = [0]
        pz = [initial_momenta]
        while sum(step) < step_goal:
            is_single_step(True)
            simulate_muon(px[-1], py[-1], pz[-1], 1, 0, 0, 0)
            data = collect()
            if len(data['step_length']) == 0:
                break
            step.append(data['step_length'][0])
            px.append(data['px'][0])
            py.append(data['py'][0])
            pz.append(data['pz'][0])
        if single_step:
            # Transform to single step and store as single values
            px, py, pz, step = transform_to_single_step(px, py, pz, step, step_goal)
            batch_data['px'].append(px)
            batch_data['py'].append(py)
            batch_data['pz'].append(pz)
            batch_data['step_length'].append(step)
        else:
            # Store as arrays for multi-step
            batch_data['px'].append(px)
            batch_data['py'].append(py)
            batch_data['pz'].append(pz)
            batch_data['step_length'].append(step)
    return batch_data

def parallel_simulations(num_sims, detector, num_processes=4, step_goal=0.05, output_file='muons_data.h5', single_step=False, initial_momenta_bounds=(10, 400)):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('initial_momenta', (0,), maxshape=(None,), dtype='f8')
        if single_step:
            # For single step, use regular float64 datasets
            f.create_dataset('px', (0,), maxshape=(None,), dtype='f8')
            f.create_dataset('py', (0,), maxshape=(None,), dtype='f8')
            f.create_dataset('pz', (0,), maxshape=(None,), dtype='f8')
            f.create_dataset('step_length', data=step_goal, dtype='f8')
        else:
            # For multi-step, use variable length arrays
            dt_vlen_f8 = h5py.special_dtype(vlen=np.float64)
            f.create_dataset('px', (0,), maxshape=(None,), dtype=dt_vlen_f8)
            f.create_dataset('py', (0,), maxshape=(None,), dtype=dt_vlen_f8)
            f.create_dataset('pz', (0,), maxshape=(None,), dtype=dt_vlen_f8)
            f.create_dataset('step_length', (0,), maxshape=(None,), dtype=dt_vlen_f8)
    simulate_fn = functools.partial(simulate_muon_batch, detector=detector, step_goal=step_goal, single_step=single_step, initial_momenta_bounds=initial_momenta_bounds)
    chunks = []
    chunk_size = min(10000, num_sims // num_processes)
    remaining = num_sims
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        chunks.append(current_chunk)
        remaining -= current_chunk
    total_written = 0
    with h5py.File(output_file, 'a') as f:
        with mp.Pool(processes=num_processes) as pool:
            for batch_result in pool.imap_unordered(simulate_fn, chunks):
                current_size = f['pz'].shape[0]
                batch_size = len(batch_result['pz'])
                new_size = current_size + batch_size
                f['px'].resize((new_size,))
                f['py'].resize((new_size,))
                f['pz'].resize((new_size,))
                f['initial_momenta'].resize((new_size,))
                f['initial_momenta'][current_size:new_size] = batch_result['initial_momenta']
                if single_step:
                    # For single step, write as regular arrays
                    f['px'][current_size:new_size] = batch_result['px']
                    f['py'][current_size:new_size] = batch_result['py']
                    f['pz'][current_size:new_size] = batch_result['pz']
                else:
                    # For multi-step, write as variable length arrays
                    f['step_length'].resize((new_size,))
                    f['px'][current_size:new_size] = [np.array(px) for px in batch_result['px']]
                    f['py'][current_size:new_size] = [np.array(py) for py in batch_result['py']]
                    f['pz'][current_size:new_size] = [np.array(pz) for pz in batch_result['pz']]
                    f['step_length'][current_size:new_size] = [np.array(step) for step in batch_result['step_length']]
                f.flush()
                total_written += batch_size
                print(f"Written batch of {batch_size} simulations. Total: {total_written}/{num_sims}")
    print(f"Completed writing {total_written} simulations to {output_file}")
    return output_file

def main(cores=16, num_sims=1_000_000, step_goal=0.05, single_step = False, initial_momenta=(10, 400), tag:str = '5'):
    folder = f"data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    detector = sphere_design.get_design()
    detector["limits"]["max_step_length"] = step_goal
    data_type = "interpolated" if single_step else "multi"
    output_file = os.path.join(folder, f"muon_data_energy_loss_{data_type}_step_{tag}.h5")
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
    parser = argparse.ArgumentParser(description="Run muon simulations.")
    parser.add_argument('--cores', type=int, default=16, help='Number of CPU cores to use')
    parser.add_argument('--num_sims', type=int, default=1_000_000, help='Number of simulations to run')
    parser.add_argument('--step_goal', type=float, default=0.02, help='Step goal for simulation')
    parser.add_argument('--split', action='store_true', help='Use main_split instead of main')
    parser.add_argument('--multi_step', dest='single_step', action='store_false', help='Return full data, without transformation to single step')
    parser.add_argument('--initial_momenta', type=float, nargs=2, default=(10, 400), help='Range of initial momenta for muons (min, max)')
    args = parser.parse_args()
    if args.split:
        main_split(cores=args.cores, num_sims=args.num_sims, step_goal=args.step_goal, single_step=args.single_step, initial_momenta=args.initial_momenta)
    else:
        main(cores=args.cores, num_sims=args.num_sims, step_goal=args.step_goal, tag=int(args.step_goal*100), single_step=args.single_step, initial_momenta=args.initial_momenta)