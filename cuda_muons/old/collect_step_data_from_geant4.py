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
from array import array

def get_energy_bin(nums):
    """
    Vectorized numpy version of the get_first_bin function.
    It calculates the bin index for each element in nums,
    based on the range 10-200, with a bin width of 10.
    If a number is outside the range, it returns -1 for that number.

    Parameters:
    nums (np.array): Numpy array of integers.

    Returns:
    np.array: Array of bin indices or -1 for out-of-range values.
    """
    # Convert input to numpy array if it isn't already
    nums = np.asarray(nums)

    # Apply the logic using numpy's vectorized operations
    bins = np.where((nums >= 10) & (nums <= 200), (nums - 10) // 10, -1)

    return bins.astype(np.int32)

def get_log_bin(num_array):
    # Convert input to numpy array if it's not already
    num_array = np.asarray(num_array)

    # Create a mask for out-of-range values
    out_of_range_mask = (num_array < -22.0) | (num_array > 0.0)

    # Calculate indices for values within range
    index_array = ((num_array + 22.0) / 0.22).astype(int)

    # Clip indices to stay within the range [0, 99]
    index_array = np.clip(index_array, 0, 99)

    # Set indices to -1 for out-of-range values
    index_array[out_of_range_mask] = -1

    return index_array.astype(np.int32)



def rotation_matrices_to_align_vectors(vectors, target_direction=np.array([0, 0, 1])):
    """
    Given an array of vectors (Nx3), return an array of rotation matrices (Nx3x3)
    that rotate each vector to align with the specified target direction.

    Parameters:
    - vectors: np.ndarray of shape (N, 3), the input vectors to align
    - target_direction: np.ndarray of shape (3,), the target direction to align each vector to (default is z-axis)

    Returns:
    - rotation_matrices: np.ndarray of shape (N, 3, 3), each matrix rotates the corresponding vector to the target direction
    """
    # Ensure the target direction is a unit vector
    target_direction = target_direction / np.linalg.norm(target_direction)

    # Normalize the input vectors to get their directions
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / norms

    # Compute the cross product and the angle for each vector
    cross_products = np.cross(directions, target_direction)
    dot_products = np.einsum('ij,j->i', directions, target_direction)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    # Prepare the rotation matrices
    N = len(vectors)
    rotation_matrices = np.zeros((N, 3, 3))

    # Iterate over each vector to compute its rotation matrix
    for i in range(N):
        angle = angles[i]
        axis = cross_products[i]
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            # If the axis norm is near zero, the vectors are aligned, no rotation needed.
            rotation_matrices[i] = np.eye(3)
        else:
            # Normalize the rotation axis
            axis /= axis_norm

            # Compute the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])

            rotation_matrices[i] = (
                    np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            )

    return rotation_matrices

# Function to simulate a batch of muons in a single process
def simulate_muon_batch(num_simulations, detector):
    # Set a unique seed based on the process ID
    np.random.seed((os.getpid() * int(time.time())) % 2**32)

    # Initialize a container for batch data
    # batch_data = {
    #     'initial_momenta': np.zeros(num_simulations),
    #     'px': np.zeros(num_simulations),
    #     'py': np.zeros(num_simulations),
    #     'pz': np.zeros(num_simulations),
    #     'step_length': np.zeros(num_simulations),
    # }
    batch_data = {
        'initial_momenta': array('f'),
        'px': array('f'),
        'py': array('f'),
        'pz': array('f'),
        'step_length': array('f'),
    }

    # Initialize and simulate muon
    initialize(np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), np.random.randint(32000), json.dumps(detector))
    z_direction = np.array([0,0,1])

    edges_ = np.linspace(-22, 0, 100 + 1)
    the_histograms = {
        'hist_dpz': np.zeros((19, 100), np.int32),
        'hist_transverse': np.zeros((19, 100), np.int32),
        'hist_dpz_x_transverse': np.zeros((19, 100, 100), np.int32),
        'hist_step_length': np.zeros((19, 100), np.int32),
        'edges_dpz': edges_ * 1.0,
        'edges_secondary': edges_ * 1.0,
        'edges_step_length': edges_ * 1.0,
        'hist_mag_x_step_length': np.zeros((19, 100, 100), np.int32),
        'hist_transverse_x_step_length': np.zeros((19, 100, 100), np.int32),
    }

    # Run the batch of simulations
    for i in range(num_simulations):
        t1 = time.time()
        initial_momenta = np.random.uniform(10, 200)

        # is_single_step(True)
        simulate_muon(0, 0, initial_momenta, 1, 0, 0, 0)
        data = collect()

        matrix_first_step = np.column_stack((data['px'][0:-1], data['py'][0:-1], data['pz'][0:-1]))

        matrix_first_step_norm = np.sqrt(np.sum(matrix_first_step**2, axis=1))
        # TODO: Maybe make it less than 100.00
        filt = matrix_first_step_norm > 10.0 # Keep it at >0.0 to prevent undefined rotation vector in rare cases the momenta turns out to be zero
        # if np.mean(filt) < 1.0:
        #     0/0
        matrix_second_step = np.column_stack((data['px'][1:], data['py'][1:], data['pz'][1:]))

        matrix_first_step = matrix_first_step[filt]
        matrix_second_step = matrix_second_step[filt]
        step_length = data['step_length'][1:][filt]
        matrix_first_step_norm = matrix_first_step_norm[filt]

        rotation_matrices = rotation_matrices_to_align_vectors(matrix_first_step, z_direction)

        matrix_first_step = np.einsum('nij,nj->ni', rotation_matrices, matrix_first_step)
        matrix_second_step = np.einsum('nij,nj->ni', rotation_matrices, matrix_second_step)

        energy_bins = get_energy_bin(matrix_first_step[:, 2])

        delta_pz_f = np.log(np.abs(matrix_second_step[:, 2] - matrix_first_step[:, 2]) / matrix_first_step[:, 2])
        delta_pz_f = np.where(np.isnan(delta_pz_f), -22, delta_pz_f)
        delta_pz_f = np.where(delta_pz_f == -np.inf, -22, delta_pz_f)
        delta_dpz_bin = get_log_bin(delta_pz_f)

        step_length_f = np.log(step_length)
        step_length_bin = get_log_bin(step_length_f)

        transverse_f = np.log(np.sqrt(matrix_second_step[:, 0] ** 2 + matrix_second_step[:, 1] ** 2) /  matrix_first_step[:, 2])
        transverse_f = np.where(np.isnan(transverse_f), -22, transverse_f)
        transverse_bin = get_log_bin(transverse_f)

        delta_mag_f = np.log(np.sqrt(matrix_second_step[:, 0] ** 2 + matrix_second_step[:, 1] ** 2+ (matrix_second_step[:, 2] - matrix_first_step[:, 2]) ** 2) / matrix_first_step[:, 2])
        delta_mag_f = np.where(np.isnan(delta_mag_f), -22, delta_mag_f)
        mag_bin = get_log_bin(delta_mag_f)

        # Fill dpz histogram
        np.add.at(the_histograms['hist_dpz'], (energy_bins, delta_dpz_bin), 1)
        np.add.at(the_histograms['hist_transverse'], (energy_bins, transverse_bin), 1)
        np.add.at(the_histograms['hist_dpz_x_transverse'], (energy_bins, delta_dpz_bin, transverse_bin), 1)
        np.add.at(the_histograms['hist_step_length'], (energy_bins, step_length_bin), 1)
        np.add.at(the_histograms['hist_step_length'], (energy_bins, step_length_bin), 1)
        np.add.at(the_histograms['hist_mag_x_step_length'], (energy_bins, mag_bin, step_length_bin), 1)
        np.add.at(the_histograms['hist_transverse_x_step_length'], (energy_bins, transverse_bin, step_length_bin), 1)

        print("Took", time.time() - t1,"seconds for", len(matrix_first_step),"steps.")

    return the_histograms

def parallel_simulations(num_sims, detector, num_processes=4):
    # Set up multiprocessing pool and run simulations
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(simulate_muon_batch, [(num_sims, detector)] * num_processes)

    # # Concatenate the results from each process
    # concatenated_data = {
    #     'initial_momenta': np.concatenate([result['initial_momenta'] for result in results]),
    #     'px': np.concatenate([result['px'] for result in results]),
    #     'py': np.concatenate([result['py'] for result in results]),
    #     'pz': np.concatenate([result['pz'] for result in results]),
    #     'step_length': np.concatenate([result['step_length'] for result in results]),
    # }

    return results


def main(num_processes=30):
    # Generate a random suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Create the file name with the random suffix
    pickle_file = f"data/multi/muon_data_energy_loss_single_step_{suffix}.pkl"
    # pickle_file = 'data/muon_data_energy_loss_single_step_p3.pkl'

    try:
        # Dump muon_data to a pickle file
        with gzip.open(pickle_file, 'wb') as f:
            pickle.dump({}, f)
        print("Written tempo for verification")
    except FileNotFoundError:
        print("Something is wrong")

    # Example usage:
    detector = sphere_design.get_design()
    # detector['limits']['max_step_length'] = 0.05
    detector['store_primary'] = True
    detector['store_all'] = False


    # Generate a random suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Create the file name with the random suffix
    pickle_file = f"data/big_no_max_lim/muon_data_energy_loss_single_step_{suffix}.pkl"
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    # pickle_file = 'data/muon_data_energy_loss_single_step_p3.pkl'

    num_sims = 4000
    stored_data = parallel_simulations(num_sims, detector, num_processes=num_processes)

    try:
        # Dump muon_data to a pickle file
        with gzip.open(pickle_file, 'wb') as f:
            pickle.dump(stored_data, f)
        print("Written")
    except FileNotFoundError:
        print("Something is wrong")


if __name__ == "__main__":
    argh.dispatch_command(main)