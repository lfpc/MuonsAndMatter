import gzip
import pickle
import time

import argh
import numpy as np
import json
import multiprocessing as mp
import os
from muon_slabs import add, simulate_muon, initialize, collect, set_field_value, set_kill_momenta, kill_secondary_tracks, is_single_step
import random
import lib.gigantic_sphere as sphere_design
import string

# Function to simulate a batch of muons in a single process
def simulate_muon_batch(num_simulations, detector):
    # Set a unique seed based on the process ID
    np.random.seed((os.getpid() * int(time.time())) % 2**32)

    # Initialize a container for batch data
    batch_data = {
        'initial_momenta': np.zeros(num_simulations),
        'px': np.zeros(num_simulations),
        'py': np.zeros(num_simulations),
        'pz': np.zeros(num_simulations),
        'step_length': np.zeros(num_simulations),
    }
    # Initialize and simulate muon
    initialize(np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), json.dumps(detector))

    # Run the batch of simulations
    for i in range(num_simulations):
        initial_momenta = np.random.uniform(10, 200)

        is_single_step(True)
        simulate_muon(0, 0, initial_momenta, 1, 0, 0, 0)
        data = collect()

        # Store the data
        batch_data['initial_momenta'][i] = initial_momenta
        batch_data['px'][i] = data['px'][0]
        batch_data['py'][i] = data['py'][0]
        batch_data['pz'][i] = data['pz'][0]
        batch_data['step_length'][i] = data['step_length'][0]

    return batch_data

def parallel_simulations(num_sims, detector, num_processes=4):
    # Set up multiprocessing pool and run simulations
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(simulate_muon_batch, [(int(num_sims/num_processes), detector)] * num_processes)

    # Concatenate the results from each process
    concatenated_data = {
        'initial_momenta': np.concatenate([result['initial_momenta'] for result in results]),
        'px': np.concatenate([result['px'] for result in results]),
        'py': np.concatenate([result['py'] for result in results]),
        'pz': np.concatenate([result['pz'] for result in results]),
        'step_length': np.concatenate([result['step_length'] for result in results]),
    }

    return concatenated_data


def main(cores=80, num_sims=10_000_000):

    detector = sphere_design.get_design()
    stored_data = parallel_simulations(num_sims, detector, num_processes=cores)

    # Generate a random suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Create the file name with the random suffix
    pickle_file = f"data/multi/muon_data_energy_loss_single_step_{suffix}.pkl"
    # pickle_file = 'data/muon_data_energy_loss_single_step_p3.pkl'

    try:
        # Dump muon_data to a pickle file
        with gzip.open(pickle_file, 'wb') as f:
            pickle.dump(stored_data, f)
        print("Written")
    except FileNotFoundError:
        print("Something is wrong")


if __name__ == "__main__":
    argh.dispatch_command(main)
