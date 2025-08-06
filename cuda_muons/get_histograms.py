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

def compute_2d_histo(array_a, array_b, edges):
    t1 = time.time()
    hist2d, _, _ = np.histogram2d(array_a, array_b, bins=[edges, edges])
    print("Took", time.time() - t1, "seconds")
    print(hist2d.shape)
    return hist2d

energy_segmentation = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

def load_h5_data(h5_filename, chunk_size=100_000):
    """
    Load data from H5 file in chunks
    """
    all_px = []
    all_py = []
    all_pz = []
    all_initial_momenta = []
    
    with h5py.File(h5_filename, 'r') as f:
        total_size = f['initial_momenta'].shape[0]
        print(f"Total records in H5 file: {total_size}")
        step_length = f['step_length'][()]
        
        # Load data in chunks
        for start_idx in range(0, total_size, chunk_size):
            end_idx = min(start_idx + chunk_size, total_size)
            print(f"Loading chunk {start_idx}:{end_idx}")
            
            chunk_initial_momenta = f['initial_momenta'][start_idx:end_idx]
            chunk_px = f['px'][start_idx:end_idx]
            chunk_py = f['py'][start_idx:end_idx] 
            chunk_pz = f['pz'][start_idx:end_idx]
            
            # Handle variable length arrays (for multi-step data)
            if hasattr(chunk_px[0], '__len__'):
                # Multi-step data - take final values
                chunk_px = np.array([px[-1] if len(px) > 0 else 0 for px in chunk_px])
                chunk_py = np.array([py[-1] if len(py) > 0 else 0 for py in chunk_py])
                chunk_pz = np.array([pz[-1] if len(pz) > 0 else pz[0] for pz in chunk_pz])
            
            all_px.extend(chunk_px)
            all_py.extend(chunk_py)
            all_pz.extend(chunk_pz)
            all_initial_momenta.extend(chunk_initial_momenta)
    
    return {
        'px': np.array(all_px),
        'py': np.array(all_py),
        'pz': np.array(all_pz),
        'initial_momenta': np.array(all_initial_momenta),
        'step_length': step_length,
    }

# Load H5 data
h5_filename = 'data/muon_data_energy_loss_interpolated_step_2.h5'
print("Loading H5 data...")
data = load_h5_data(h5_filename)

px = data['px']
py = data['py']
pz = data['pz']
initial_momenta = data['initial_momenta']

print(f"Loaded {len(px)} records")

# Filtered values for delta_pz_f and delta_px_f
delta_pz_f = np.log(np.abs((pz - initial_momenta) / initial_momenta))
delta_pz_f = np.where(np.isnan(delta_pz_f), -20, delta_pz_f)
delta_pz_f = np.where(delta_pz_f == float('-inf'), -20, delta_pz_f)


secondary_f = np.log(np.sqrt(px**2 + py**2) / initial_momenta)
secondary_f = np.where(np.isnan(secondary_f), -20, secondary_f)

nbins = 100

# Dynamic edges
edges_dpz = np.linspace(np.min(delta_pz_f), np.max(delta_pz_f), nbins + 1)
edges_secondary = np.linspace(np.min(secondary_f), np.max(secondary_f), nbins + 1)

# Static edges (uncomment if preferred)
# edges_dpz = np.linspace(-22, 0, nbins + 1)
# edges_secondary = np.linspace(-22, 0, nbins + 1)

# Convert each array to a comma-separated string
string1 = ','.join(map(str, edges_dpz))
string3 = ','.join(map(str, edges_secondary))

# Write the strings to a text file
with open("plots/edges/edges.txt", "w") as file:
    file.write("edges z: " + string1 + "\n")
    file.write("edges secondary: " + string3 + "\n")

hists_dict = {}

for i in range(len(energy_segmentation)-1):
    print(f"Processing energy segment {energy_segmentation[i]} - {energy_segmentation[i+1]} GeV")
    
    # Create a filter for the current segmentation
    filt = (initial_momenta >= energy_segmentation[i]) & (initial_momenta <= energy_segmentation[i + 1])
    
    if np.sum(filt) == 0:
        print(f"No data in energy range {energy_segmentation[i]} - {energy_segmentation[i+1]} GeV")
        continue
    
    # Filtered values for delta_pz_f and delta_px_f
    delta_pz_f_filt = np.log(np.abs((pz[filt] - initial_momenta[filt]) / initial_momenta[filt]))
    delta_pz_f_filt = np.where(np.isnan(delta_pz_f_filt), -22, delta_pz_f_filt)
    delta_pz_f_filt = np.where(delta_pz_f_filt == float('-inf'), -22, delta_pz_f_filt)
    
    
    secondary_f_filt = np.log(np.sqrt(px[filt]**2 + py[filt]**2) / initial_momenta[filt])
    secondary_f_filt = np.where(np.isnan(secondary_f_filt), -22, secondary_f_filt)
    
    total_loss_mag = np.log(np.sqrt(px[filt]**2 + py[filt]**2 + (pz[filt] - initial_momenta[filt])**2) / initial_momenta[filt])
    
    # Use static edges
    edges_dpz = np.linspace(-22, 0, nbins + 1)
    edges_secondary = np.linspace(-22, 0, nbins + 1)
    
    # Compute histograms
    hist_dpz, _ = np.histogram(delta_pz_f_filt, bins=edges_dpz)
    hist_secondary, _ = np.histogram(secondary_f_filt, bins=edges_secondary)
    
    hist_2d = compute_2d_histo(delta_pz_f_filt, secondary_f_filt, edges_dpz)
    
    hists_dict[(energy_segmentation[i], energy_segmentation[i + 1])] = {
        #'hist_dpz': hist_dpz,
        #'hist_secondary': hist_secondary,
        'step_length': data['step_length'], 
        'hist_2d': hist_2d,
        'edges_dpz': edges_dpz,
        'edges_secondary': edges_secondary,
    }
    
    plotting_enabled = True
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
        
        ax[0].stairs(hist_dpz, edges_dpz, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
        ax[1].stairs(hist_secondary, edges_secondary, label='%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
        
        ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
        ax[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
        
        ax[0].set_ylabel('Frequency (arb.)')
        ax[1].set_ylabel('Frequency (arb.)')
        
        ax[0].set_xlabel(r'$\log\left(\frac{\Delta p_z}{\text{initial } p_z}\right)$')
        ax[1].set_xlabel(r'$\log\left(\frac{\sqrt{p_x^2 + p_y^2}}{\text{initial } p_z}\right)$')
        
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        
        print("Plotting...")
        plt.savefig(f'plots/hists_a/{energy_segmentation[i]}-{energy_segmentation[i+1]}_dpz.pdf', bbox_inches='tight')
        plt.close()

# Save histograms
with open("data/histograms.pkl", "wb") as file:
    pickle.dump(hists_dict, file)
print("Histograms built and saved.")