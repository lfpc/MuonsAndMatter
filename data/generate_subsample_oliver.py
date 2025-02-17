import gzip
import numpy as np
import pickle
from tqdm import tqdm

# Define input and output files
input_file = "full_sample/full_sample_0.pkl"  # Input pickle file (gzip compressed)
output_file = "enriched_input.pkl"

# Load data from the pickle file
with gzip.open(input_file, 'rb') as f:
    data = pickle.load(f)

# Extract px, py, pz from the loaded data
px,py,pz,x,y,z,pdg,weight = data.T

# Initialize 2D histogram
hist, xedges, yedges = np.histogram2d([], [], bins=[100, 100], range=[[0, 350], [0, 6]])

# Arrays to store the data to be pickled
data_to_pickle = []

# First pass: Fill the histogram and select entries
p = np.sqrt(px**2 + py**2 + pz**2)
pt = np.sqrt(px**2 + py**2)

b_x = (p / 350.0 * 100).astype(int) + 1
b_y = (pt / 6.0 * 100).astype(int) + 1

for i in tqdm(range(len(px))):
    # Clamp bin indices to avoid out-of-bounds
    bx = min(max(b_x[i], 0), 99)
    by = min(max(b_y[i], 0), 99)
    if hist[bx, by] < 100:
        hist[bx, by] += 1
        data_to_pickle.append([px[i], py[i], pz[i], x[i], y[i], z[i], pdg[i], weight[i]])

data_to_pickle = np.array(data_to_pickle)

# Second pass: Modify and fill additional entries
additional_data = []

# Use selected data from the first pass
px_sel, py_sel, pz_sel, x, y, z, pdg, weight = data_to_pickle.T

p_sel = np.sqrt(px_sel**2 + py_sel**2 + pz_sel**2)
pt_sel = np.sqrt(px_sel**2 + py_sel**2)

b_x_sel = (p_sel / 350.0 * 100).astype(int) + 1
b_y_sel = (pt_sel / 6.0 * 100).astype(int) + 1

for i in tqdm(range(len(px_sel))):
    bx = min(max(b_x_sel[i], 0), 99)
    by = min(max(b_y_sel[i], 0), 99)
    if hist[bx, by] < 10:
        n = int(10 / hist[bx, by]) if hist[bx, by] > 0 else 10
        for _ in range(n):
            phi = np.random.uniform(0, 2 * np.pi)
            new_px = pt_sel[i] * np.cos(phi)
            new_py = pt_sel[i] * np.sin(phi)
            additional_data.append([new_px, new_py, pz_sel[i], x[i], y[i], z[i], pdg[i], weight[i]])
    additional_data.append([pt_sel[i], 0, pz_sel[i], x[i], y[i], z[i], pdg[i], weight[i]])
    additional_data.append([-pt_sel[i], 0, pz_sel[i], x[i], y[i], z[i], pdg[i], weight[i]])

# Combine initial and additional data
additional_data = np.array(additional_data)
all_data = np.concatenate((data_to_pickle, additional_data), axis=0)

# Save the data to a pickle file
with gzip.open(output_file, 'wb') as f:
    pickle.dump(all_data, f)
print("Data SHAPE", all_data.shape)
print("Data has been written to", output_file)