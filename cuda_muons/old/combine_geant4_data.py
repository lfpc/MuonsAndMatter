import os
import gzip
import pickle
import numpy as np
from tqdm import tqdm
import argparse


def load_pickled_data(file_path):
    """Loads data from a gzipped pickle file."""
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def concatenate_results(data_list):
    """Concatenates data from multiple dictionaries."""
    concatenated_data = {
        'initial_momenta': [data['initial_momenta'] for data in data_list],
        'px': [data['px'] for data in data_list],
        'py': [data['py'] for data in data_list],
        'pz': [data['pz'] for data in data_list],
        'step_length': [data['step_length'] for data in data_list],
    }
    return concatenated_data


def save_combined_data(file_path, data):
    """Saves concatenated data into a gzipped pickle file."""
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)


def main(folder_path):
    # Collect all .pkl files in the specified folder
    pkl_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pkl')][:5]

    all_data = [load_pickled_data(file) for file in tqdm(pkl_files, desc="Loading data files")]
    print('Loaded data from {} files'.format(len(all_data)))

    # Concatenate all data
    combined_data = concatenate_results(all_data)

    # Save combined data to joined.pkl
    output_file = os.path.join(folder_path, 'joined.pkl')
    save_combined_data(output_file, combined_data)
    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    # Argument parser for command line input
    parser = argparse.ArgumentParser(description="Combine Geant4 data from multiple pickle files.")
    parser.add_argument('-dir', type=str, help="Path to the folder containing .pkl files", default='multi')
    args = parser.parse_args()

    folder_path = '/home/hep/lprate/projects/cuda_muons/cuda_muons/data'
    folder_path = os.path.join(folder_path, args.dir)

    main(folder_path)

