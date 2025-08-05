import gzip
import numpy as np
import pickle
from tqdm import tqdm
import os
import h5py

output_file = "subsample_40M.h5"
N_subsamples = 40E6
N_total = 40E6

input_files = os.listdir('../full_sample')
num_per_file = int(N_subsamples / len(input_files))
with h5py.File(output_file, 'w') as out_f:
    out_f.create_dataset('px', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('py', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('pz', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('x', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('y', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('z', shape=(N_subsamples,), dtype='f4')
    out_f.create_dataset('pdg', shape=(N_subsamples,), dtype='i4')
    out_f.create_dataset('weight', shape=(N_subsamples,), dtype='f4')
    print(f"Output file will contain {N_subsamples} entries, {num_per_file} per input file.")
    for i,file_name in enumerate(input_files):
        print(f"Processing file {i+1}/{len(input_files)}: {file_name}")
        with h5py.File(f'../full_sample/{file_name}', 'r') as f:
            px = f['px']
            py = f['py']
            pz = f['pz']
            x = f['x']
            y = f['y']
            z = f['z']
            pdg = f['pdg']
            weight = f['weight']
            num_rows = px.shape[0]
            start_idx = i * num_per_file
            end_idx = start_idx + num_per_file
            out_f['px'][start_idx:end_idx] = px[:num_per_file]
            out_f['py'][start_idx:end_idx] = py[:num_per_file]
            out_f['pz'][start_idx:end_idx] = pz[:num_per_file]
            out_f['x'][start_idx:end_idx] = x[:num_per_file]
            out_f['y'][start_idx:end_idx] = y[:num_per_file]
            out_f['z'][start_idx:end_idx] = z[:num_per_file]
            out_f['pdg'][start_idx:end_idx] = pdg[:num_per_file]
            out_f['weight'][start_idx:end_idx] = weight[:num_per_file]
        print(f"Processed file {i+1}/{len(input_files)}: {file_name}, rows: {num_rows}, subsampled: {num_per_file}")

print("Data has been written to", output_file)