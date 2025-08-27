import time
import pickle
import torch


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


histogram_file='data/histograms.pkl'
save_dir = 'data/alias_histograms.pkl'

with open(histogram_file, 'rb') as f:
    stored_hist_data_all = pickle.load(f)

step_length = stored_hist_data_all.pop('step_length')
energy_bins = list(stored_hist_data_all.keys())
keys = stored_hist_data_all[energy_bins[0]].keys()
n_energy_bins = len(energy_bins)
print(f"Processing {n_energy_bins} energy bins vectorized")
stored_hist_data = {key:torch.tensor([stored_hist_data_all[energy_bin][key].tolist() for energy_bin in energy_bins]) for key in keys}

edges_dpz = stored_hist_data['edges_dpz'][0]
edges_secondary = stored_hist_data['edges_secondary'][0]
centers_dpz = (edges_dpz[:-1] + edges_dpz[1:]) / 2
centers_secondary = (edges_secondary[:-1] + edges_secondary[1:]) / 2
widths_dpz = edges_dpz[1:] - edges_dpz[:-1]
widths_secondary = edges_secondary[1:] - edges_secondary[:-1]
hist_2d_all = stored_hist_data['hist_2d'].reshape((-1))

n_bins_dpz = len(centers_dpz)
n_bins_secondary = len(centers_secondary)
assert n_bins_dpz == n_bins_secondary
hist_2d_bin_centers_first_dim = centers_dpz.repeat_interleave(n_bins_secondary)
hist_2d_bin_centers_second_dim = centers_secondary.repeat(n_bins_dpz)
hist_2d_bin_widths_first_dim = widths_dpz.repeat_interleave(n_bins_secondary)
hist_2d_bin_widths_second_dim = widths_secondary.repeat(n_bins_dpz)

hist_2d_all = hist_2d_all.reshape((-1,100*100))
hist_2d_all = torch.nn.functional.normalize(hist_2d_all, p=1, dim=1)


t0 = time.time()
hist_2d_probability_table, hist_2d_alias_table = alias_setup(hist_2d_all)
print(f"Alias setup took {time.time() - t0:.4f} seconds")

output = {
    'hist_2d_probability_table': hist_2d_probability_table,
    'hist_2d_alias_table': hist_2d_alias_table,
    'centers_first_dim': hist_2d_bin_centers_first_dim,
    'centers_second_dim': hist_2d_bin_centers_second_dim,
    'width_first_dim': hist_2d_bin_widths_first_dim,
    'width_second_dim': hist_2d_bin_widths_second_dim,
    'step_length': step_length
}
for k, v in output.items():
    if k == 'step_length':
        print(f"{k}: value={v}")
    else:
        print(f"{k}: shape={v.shape}")

with open(save_dir, 'wb') as f:
    pickle.dump(output, f)
print("Data saved to", save_dir)
