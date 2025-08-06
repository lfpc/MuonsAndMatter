import os.path
import time

import numpy as np
import pickle
import gzip
import torch
assert torch.cuda.is_available(), f"CUDA is not available. Torch version: {torch.__version__} \n Torch cuda version: {print(torch.version.cuda)}"

#from helpers_cuda_muons import get_sample_arb8s

import faster_muons_torch
server = True


from tqdm import tqdm

def alias_setup(histogram):
    N = histogram.shape[0]
    n = histogram.shape[1]
    prob_table = torch.zeros_like(histogram)
    alias_table = torch.zeros(histogram.shape, dtype=torch.int32)

    # Normalize probabilities and scale by n
    normalized_prob = histogram*n

    for j in range(N):
        small = []
        large = []

        # Separate bins into small and large
        for i, p in enumerate(normalized_prob[j]):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        # Distribute probabilities between small and large bins
        while small and large:
            small_bin = small.pop()
            large_bin = large.pop()

            prob_table[j][small_bin] = normalized_prob[j][small_bin]
            alias_table[j][small_bin] = large_bin

            # Adjust the large bin's probability
            normalized_prob[j][large_bin] = (normalized_prob[j][large_bin] +
                                          normalized_prob[j][small_bin] - 1)
            if normalized_prob[j][large_bin] < 1:
                small.append(large_bin)
            else:
                large.append(large_bin)

        # Fill remaining bins
        for remaining in large + small:
            prob_table[j][remaining] = 1
            alias_table[j][remaining] = remaining

    return prob_table, alias_table

def propagate_muons_with_cuda(
    muons_positions,
    muons_momenta,
    hist_2d_probability_table,
    hist_2d_alias_table,
    hist_2d_bin_centers_first_dim,
    hist_2d_bin_centers_second_dim,
    hist_2d_bin_widths_first_dim,
    hist_2d_bin_widths_second_dim,
    kill_at=-1.0,
    num_steps=100,
    step_length_fixed=0.02,
    seed=1234,
):

    # Ensure inputs are float tensors on CUDA
    muons_positions_cuda = muons_positions.float().cuda()
    muons_momenta_cuda = muons_momenta.float().cuda()
    hist_2d_probability_table_cuda = hist_2d_probability_table.float().cuda()
    hist_2d_alias_table_cuda = hist_2d_alias_table.int().cuda()
    hist_2d_bin_centers_first_dim_cuda = hist_2d_bin_centers_first_dim.float().cuda()
    hist_2d_bin_centers_second_dim_cuda = hist_2d_bin_centers_second_dim.float().cuda()
    hist_2d_bin_widths_first_dim_cuda = hist_2d_bin_widths_first_dim.float().cuda()
    hist_2d_bin_widths_second_dim_cuda = hist_2d_bin_widths_second_dim.float().cuda()

    magnetic_field = torch.zeros((100,100,1000, 3), dtype=torch.float32).cuda()
    magnetic_field[:,:,:,1] = 1.0
    magnetic_field_ranges = torch.from_numpy(np.zeros((1, 6))).float().cpu()

    #arb8s = get_sample_arb8s()
    arb8s = torch.ones((8,5)).cuda()

    t1 = time.time()
    # Call the function
    # Check device of all inputs except the last 3

    # Build simple uniform grid for Arb8 lookup
    # Example: resolution and bounds match magnetic_field_ranges
    gx, gy, gz = 10, 10, 10
    arb8_grid_range_np = magnetic_field_ranges  # [minX,maxX,minY,maxY,minZ,maxZ]
    # empty grid: no Arb8s assigned
    total_cells = gx * gy * gz
    arb8_grid_index = torch.zeros(total_cells + 1, dtype=torch.int32, device = 'cuda')  # start offsets per cell
    arb8_grid_list = torch.zeros(0, dtype=torch.int32)                # flat list of Arb8 indices

    t1 = time.time()
    # Call the function with grid and dims
    faster_muons_torch.propagate_muons_with_alias_sampling(
        muons_positions_cuda,
        muons_momenta_cuda,
        hist_2d_probability_table_cuda,
        hist_2d_alias_table_cuda,
        hist_2d_bin_centers_first_dim_cuda,
        hist_2d_bin_centers_second_dim_cuda,
        hist_2d_bin_widths_first_dim_cuda,
        hist_2d_bin_widths_second_dim_cuda,
        magnetic_field,
        magnetic_field_ranges,
        arb8s,
        arb8_grid_index,
        arb8_grid_range_np.float(),
        # gx, gy, gz, # Removed to match the outdated compiled extension's expected signature
        kill_at,
        num_steps,
        step_length_fixed, 
        seed
    )
    torch.cuda.synchronize()
    print("Took", time.time() - t1, "seconds for %.2e muons and %d steps." % (len(muons_positions_cuda), num_steps))

    # Convert results back to numpy arrays and return
    return muons_positions_cuda.cpu().numpy(), muons_momenta_cuda.cpu().numpy()

def run(muons:np.array, 
        histogram_file='data/combined_histograms.pkl',
        save_dir = 'data/cuda_muons_data.pkl',
        n_steps=500):
    muons = torch.from_numpy(muons).float()
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]

    with open(histogram_file, 'rb') as f:
        stored_hist_data_all = pickle.load(f)

    energy_bins = list(stored_hist_data_all.keys())
    keys = stored_hist_data_all[energy_bins[0]].keys()
    n_energy_bins = len(energy_bins)

    print(f"Processing {n_energy_bins} energy bins vectorized")
    

    stored_hist_data = {key:torch.tensor([stored_hist_data_all[energy_bin][key].tolist() for energy_bin in energy_bins]) for key in keys}
    step_length_fixed = float(stored_hist_data['step_length'][0])
    edges_dpz = stored_hist_data['edges_dpz'][0]  # Assuming all energy bins have the same edges
    centers_full = (edges_dpz[:-1] + edges_dpz[1:]) / 2
    widths_full = edges_dpz[1:] - edges_dpz[:-1]
    hist_2d_all = stored_hist_data['hist_2d']

    n_bins = len(centers_full)
    hist_2d_bin_centers_first_dim = centers_full.repeat_interleave(n_bins)
    hist_2d_bin_centers_second_dim = centers_full.repeat(n_bins)
    hist_2d_bin_widths_first_dim = widths_full.repeat_interleave(n_bins)
    hist_2d_bin_widths_second_dim = widths_full.repeat(n_bins)

    hist_2d_all = hist_2d_all.reshape((-1,100*100))
    hist_2d_all = torch.nn.functional.normalize(hist_2d_all, p=1, dim=1)
    t0 = time.time()
    hist_2d_probability_table, hist_2d_alias_table = alias_setup(hist_2d_all)
    print(f"Alias setup took {time.time() - t0:.4f} seconds")

    print("Using CUDA for propagation... (server)")

    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            hist_2d_probability_table,
            hist_2d_alias_table,
            hist_2d_bin_centers_first_dim,
            hist_2d_bin_centers_second_dim,
            hist_2d_bin_widths_first_dim,
            hist_2d_bin_widths_second_dim,
            -1.0,
            n_steps,
            step_length_fixed,
            200
        )
    
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2]   
    }
    if save_dir is not None:
        with gzip.open(save_dir, 'wb') as f:
            pickle.dump(output, f)
        print("Data saved to", save_dir)
    return output  





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--f',dest = 'histogram_file', type=str, default='data/histograms.pkl',
                        help='Path to the histogram file')
    parser.add_argument('--n', type=int, default=500000,
                        help='Number of muons to simulate')
    args = parser.parse_args()
    
    initial_momenta = np.array([[0.,0.,170.]])
    initial_positions = np.array([[0.,0.,0.]])
    muons = np.concatenate((initial_momenta, initial_positions), axis=1)*np.ones((args.n,1))

    run(muons, histogram_file=args.histogram_file)