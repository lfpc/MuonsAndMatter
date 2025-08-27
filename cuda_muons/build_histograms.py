import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import os

from matplotlib.lines import Line2D

# Ensure all necessary directories exist
directories = [
    "data/multi",
    "plots/edges", 
    "plots/hists_a"
]

for directory in directories:
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

def compute_2d_histo(array_a, array_b, edges_a, edges_b):
    t1 = time.time()
    hist2d, _, _ = np.histogram2d(array_a, array_b, bins=[edges_a, edges_b])
    print("Took", time.time() - t1, "seconds")
    print(hist2d.shape)
    return hist2d


# Load H5 data
h5_filename = "data/muon_data_energy_loss_sens_step_2.h5"
plotting_enabled = True

nbins = 100
min_bin = -20

# Dynamic edges#
#edges_dpz = np.linspace(np.min(delta_pz_f), np.max(delta_pz_f), nbins + 1)
#edges_secondary = np.linspace(np.min(secondary_f), np.max(secondary_f), nbins + 1)

# Static edges (uncomment if preferred)
edges_dpz = np.linspace(-10, 0, nbins + 1) #-10
edges_secondary = np.linspace(-18, 0, nbins + 1) #-18

hists_dict = {}
with h5py.File(h5_filename, 'r') as f:
    step_length = f.attrs['step_length']/100
    hists_dict['step_length'] = step_length
    for energy_seg in sorted(
        f.keys(),
        key=lambda k: int(k.strip("()").split(",")[0])):

        print(f"Processing energy segment {energy_seg} GeV")
        px = f[energy_seg]['px'][:]
        py = f[energy_seg]['py'][:]
        pz = f[energy_seg]['pz'][:]
        initial_momenta = f[energy_seg]['initial_momenta'][:]

        # Filtered values for delta_pz_f and delta_px_f
        delta_pz_f_filt = np.log(np.abs((pz - initial_momenta) / initial_momenta))
        delta_pz_f_filt = np.where(np.isnan(delta_pz_f_filt), min_bin, delta_pz_f_filt)
        delta_pz_f_filt = np.where(delta_pz_f_filt == float('-inf'), min_bin, delta_pz_f_filt)
    
        secondary_f_filt = np.log(np.sqrt(px**2 + py**2) / initial_momenta)
        secondary_f_filt = np.where(np.isnan(secondary_f_filt), min_bin, secondary_f_filt)
    
        total_loss_mag = np.log(np.sqrt(px**2 + py**2 + (pz - initial_momenta)**2) / initial_momenta)
    
        # Compute histograms
        hist_dpz, _ = np.histogram(delta_pz_f_filt, bins=edges_dpz)
        hist_secondary, _ = np.histogram(secondary_f_filt, bins=edges_secondary)

        hist_2d = compute_2d_histo(delta_pz_f_filt, secondary_f_filt, edges_dpz, edges_secondary)

        energy_seg = tuple(int(x.strip()) for x in energy_seg.strip("()").split(","))
        hists_dict[energy_seg] = {
            #'hist_dpz': hist_dpz,
            #'hist_secondary': hist_secondary,
            'hist_2d': hist_2d,
            'edges_dpz': edges_dpz,
            'edges_secondary': edges_secondary
        }
    
        if plotting_enabled:
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
            fig.subplots_adjust(
                top=0.9,
                bottom=0.1,
                left=0.1,
                right=0.9,
                hspace=0.35,
                wspace=0.4
            )
            ax[0].grid(True)
            ax[1].grid(True)
            
            fig.suptitle('        ')
            
            ax[0].stairs(hist_dpz, edges_dpz, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_seg[0], energy_seg[1]), color='firebrick', zorder=5)
            ax[1].stairs(hist_secondary, edges_secondary, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_seg[0], energy_seg[1]), color='firebrick', zorder=5)

            ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
            ax[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
            
            ax[0].set_ylabel('Frequency (arb.)')
            ax[1].set_ylabel('Frequency (arb.)')
            
            ax[0].set_xlabel(r'$\log\left(\frac{\Delta p_z}{\text{initial } p_z}\right)$')
            ax[1].set_xlabel(r'$\log\left(\frac{\sqrt{p_x^2 + p_y^2}}{\text{initial } p_z}\right)$')
            
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            
            print("Plotting...")
            plt.savefig(f'plots/hists_a/{energy_seg[0]}-{energy_seg[1]}_dpz.pdf', bbox_inches='tight')
            plt.close()

# Save histograms
with open("data/histograms.pkl", "wb") as file:
    pickle.dump(hists_dict, file)
print("Histograms built and saved.")


