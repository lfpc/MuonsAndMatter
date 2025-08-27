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

def iron_block(sensitive_film):
    corners = [-25., -25.,
               25., -25.,
               25.,  25.,
              -25.,  25.,
              -25., -25.,
               25., -25.,
               25.,  25.,
              -25.,  25.]
    shield = {
        "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 51, "worldSizeY": 51,
        "worldSizeZ": 101,
        "type" : 1,
        "limits" : {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        },
        "global_field_map" : {},
        "sensitive_film": sensitive_film,
        "magnets": [{"material": "G4_Fe",
                     "dz": 25.0,
                     "z_center": 25.0,
                     'components': [{
            "corners": corners,
            "field": [0.0, 0.0, 0.0],
            "field_profile": 'uniform',
            "z_center": 25.0,
            "dz": 25.0,
            'name': 'Magnet1',
            
        }]}]
        }
    return shield

# Function to simulate a batch of muons in a single
def simulate_muon_batch(muons, detector):
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
    for m in muons:
        simulate_muon(*m[:3], 1, *m[3:])
        data = collect()#_from_sensitive()
        assert False, (m,data)
        index = 0
        if len(data['pz'])>0 and 13 in np.abs(data['pdg_id']): 
            while int(abs(data['pdg_id'][index])) != 13:
                index += 1
        else: continue
        batch_data['px'].append(data['px'][index])
        batch_data['py'].append(data['py'][index])
        batch_data['pz'].append(data['pz'][index])
        i += 1
        assert False, batch_data
    return batch_data

def parallel_simulations(muons, detector, num_processes=64):
    simulate_fn = functools.partial(simulate_muon_batch, detector=detector)
    muons_chunks = np.array_split(muons, num_processes)

    results = {'initial_momenta': [],
               'px': [],
               'py': [],
               'pz': []
               }
    with mp.Pool(processes=num_processes) as pool:
        for batch_result in pool.imap_unordered(simulate_fn, muons_chunks):
            results['initial_momenta'].extend(batch_result['initial_momenta'])
            results['px'].extend(batch_result['px'])
            results['py'].extend(batch_result['py'])
            results['pz'].extend(batch_result['pz'])
    return results

def main(muons, cores, sens_plane):
    folder = f"data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    sensitive_film = {"name": "SensitiveFilm", "z_center": 10, "dz": 0.001, "dx": 20, "dy": 20}
    detector = iron_block(sensitive_film)
    detector["store_primary"] = False
    detector["store_all"] = False
    results_inside = parallel_simulations(muons, detector, num_processes=cores)
    results_penetrating = parallel_simulations(muons, detector, num_processes=cores)
    return results_inside, results_penetrating

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run muon simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=16, help='Number of CPU cores to use')
    parser.add_argument('--num_sims', type=int, default=1_000_000, help='Number of simulations to run')
    parser.add_argument('--initial_momenta', type=float, nargs=2, default=(10, 300), help='Range of initial momenta for muons (min, max)')
    parser.add_argument('--sens_plane', type=float, default=10, help='Position of the sensitive plane')
    args = parser.parse_args()

    muons = np.zeros((args.num_sims,6))
    muons[:, 2] = np.random.uniform(args.initial_momenta[0], args.initial_momenta[1], args.num_sims)
    results_inside, results_penetrating = main(muons=muons, cores=args.cores, sens_plane=args.sens_plane)
    import matplotlib.pyplot as plt

    # Calculate pt for both results
    pt_inside = np.sqrt(np.array(results_inside['px'])**2 + np.array(results_inside['py'])**2)
    pt_penetrating = np.sqrt(np.array(results_penetrating['px'])**2 + np.array(results_penetrating['py'])**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: pz distributions
    axes[0].hist(results_inside['pz'], bins=100, alpha=0.7, label='Inside', color='tab:blue', density=True)
    axes[0].hist(results_penetrating['pz'], bins=100, alpha=0.7, label='Penetrating', color='tab:orange', density=True)
    axes[0].set_xlabel('pz')
    axes[0].set_ylabel('Density')
    axes[0].set_title('pz Distribution')
    axes[0].legend()

    # Second plot: pt distributions
    axes[1].hist(pt_inside, bins=100, alpha=0.7, label='Inside', color='tab:blue', density=True)
    axes[1].hist(pt_penetrating, bins=100, alpha=0.7, label='Penetrating', color='tab:orange', density=True)
    axes[1].set_xlabel('pt = sqrt(px^2 + py^2)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('pt Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("muon_distributions.png")
    plt.close()
