import gzip
import numpy as np
import pickle
from tqdm import tqdm
import os

output_file = "subsample_4M.pkl"
N_subsamples = 4E6

# Extract px, py, pz from the loaded data

data = []
for file_name in os.listdir('full_sample'):
    with gzip.open(f'full_sample/{file_name}', 'rb') as f:
        data.append(pickle.load(f))
data = np.concatenate(data)
np.random.shuffle(data)
print('Full data shape:',data.shape)
weight = data[:,-1]
N_samples = weight.sum()
print('N_samples:',N_samples)
unique_weight,n_w = np.unique(weight,return_counts=True)
total_w = unique_weight*n_w
n_per_w = total_w/N_samples

subsample = []
for w_val,n in zip(unique_weight,n_per_w):
    print('w:',w_val)
    if w_val < 0.001: continue
    mask = weight == w_val
    n = int(n*N_subsamples)
    print('n:',n)
    subsample.append(data[mask][np.random.choice(np.sum(mask), n, replace=False)])

subsample = np.concatenate(subsample)
subsample[:,-1] = np.ones(subsample.shape[0])

# Save the data to a pickle file
with gzip.open(output_file, 'wb') as f:
    pickle.dump(subsample, f)
print("Data SHAPE", subsample.shape)
print("Data has been written to", output_file)




