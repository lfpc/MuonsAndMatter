import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def filter_data(px, py, pz, x, y, z, weights = None, propagate_until=None):
    """Filters the data based on momentum and sensor plane conditions."""
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    
    # Apply momentum filter
    mask = (p_mag >= args.filter_p[0]) & (p_mag <= args.filter_p[1])
    
    # Apply sensor plane filter if enabled
    if args.sens_plane:
        mask &= (np.abs(x) <= 2) & (np.abs(y) <= 3)
        
    px,py,pz,x,y,z = px[mask], py[mask], pz[mask], x[mask], y[mask], z[mask]
    if weights is not None:
        weights = weights[mask]
    
    if propagate_until is not None:
        x, y, z = propagate_muons_until_z(px, py, pz, x, y, z, propagate_until)
        if args.sens_plane:
            mask = (np.abs(x) <= 2) & (np.abs(y) <= 3)
            px,py,pz,x,y,z = px[mask], py[mask], pz[mask], x[mask], y[mask], z[mask]
            if weights is not None:
                weights = weights[mask]

    return px, py, pz, x, y, z, weights
    

def propagate_muons_until_z(px,py,pz,x,y,z, z_target:float):
    """ Propagate muons in a straight line until they reach z_target """
    dz = z_target - z
    y += dz*py/pz
    x += dz*px/pz
    z = np.ones_like(z)*z_target
    return x,y,z

def plot_histograms(output_filename, 
                    px_1, py_1, pz_1, x_1, y_1, z_1,
                    px_2, py_2, pz_2, x_2, y_2, z_2, weights_1 = None, weights_2 = None,
                    label_1='data1', label_2='data2'):


    
    p_mag_1 = np.sqrt(px_1 ** 2 + py_1 ** 2 + pz_1 ** 2)
    p_mag_2 = np.sqrt(px_2 ** 2 + py_2 ** 2 + pz_2 ** 2)
    print("=" * 60)
    print(f"Number of {label_1} samples after filter: {px_1.shape[0]}")
    if weights_1 is not None:
        print(f"Sum of {label_1} weights after filter: {np.sum(weights_1)}")
    print(f"Number of {label_2} samples after filter: {px_2.shape[0]}")
    if weights_2 is not None:
        print(f"Sum of {label_2} weights after filter: {np.sum(weights_2)}")

    print(f"Difference in number of muons in output:", np.abs(px_1.shape[0] - px_2.shape[0]) / px_1.shape[0] * 100, "%")

    p_transverse_1 = np.sqrt(px_1 ** 2 + py_1 ** 2)
    p_transverse_2 = np.sqrt(px_2 ** 2 + py_2 ** 2)

    num_bins = 100
    bins_px = np.linspace(min(px_1.min(), px_2.min()), max(px_1.max(), px_2.max()), num_bins)
    bins_py = np.linspace(min(py_1.min(), py_2.min()), max(py_1.max(), py_2.max()), num_bins)
    bins_pz = np.linspace(min(pz_1.min(), pz_2.min()), max(pz_1.max(), pz_2.max()), num_bins)
    bins_p_mag = np.linspace(min(p_mag_1.min(), p_mag_2.min()), max(p_mag_1.max(), p_mag_2.max()), num_bins)
    bins_p_transverse = np.linspace(min(p_transverse_1.min(), p_transverse_2.min()), max(p_transverse_1.max(), p_transverse_2.max()), num_bins)
    bins_x = np.linspace(min(x_1.min(), x_2.min()), max(x_1.max(), x_2.max()), num_bins)
    bins_y = np.linspace(min(y_1.min(), y_2.min()), max(y_1.max(), y_2.max()), num_bins)
    bins_z = np.linspace(min(z_1.min(), z_2.min()), max(z_1.max(), z_2.max()), num_bins)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    density = args.density

    ylabel = 'Density' if density else 'Count'

    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    hist_1, bins_1, _ = axs[0, 0].hist(px_1, bins=bins_px, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[0, 0].hist(px_2, bins=bins_px, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[0, 0].set_title(r'$p_x$ Component', fontsize=title_fontsize)
    axs[0, 0].set_xlabel(r'$p_x$', fontsize=label_fontsize)
    axs[0, 0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 0].legend(fontsize=legend_fontsize)



    hist_1, bins_1, _ = axs[0, 1].hist(py_1, bins=bins_py, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[0, 1].hist(py_2, bins=bins_py, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[0, 1].set_title(r'$p_y$ Component', fontsize=title_fontsize)
    axs[0, 1].set_xlabel(r'$p_y$', fontsize=label_fontsize)
    axs[0, 1].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 1].legend(fontsize=legend_fontsize)


    hist_1, bins_1, _ = axs[0, 2].hist(pz_1, bins=bins_pz, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[0, 2].hist(pz_2, bins=bins_pz, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[0, 2].set_title(r'$p_z$ Component', fontsize=title_fontsize)
    axs[0, 2].set_xlabel(r'$p_z$', fontsize=label_fontsize)
    axs[0, 2].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0, 2].legend(fontsize=legend_fontsize)

    hist_1, bins_1, _ = axs[1, 0].hist(p_mag_1, bins=bins_p_mag, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[1, 0].hist(p_mag_2, bins=bins_p_mag, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[1, 0].set_title(r'Momentum Magnitude $(|p|)$')
    axs[1, 0].set_xlabel(r'$|p|$')
    axs[1, 0].set_ylabel(ylabel)
    axs[1, 0].legend()


    hist_1, bins_1, _ = axs[1, 1].hist(p_transverse_1, bins=bins_p_transverse, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[1, 1].hist(p_transverse_2, bins=bins_p_transverse, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[1, 1].set_title(r'Transverse Momentum $(\sqrt{p_x^2 + p_y^2})$')
    axs[1, 1].set_xlabel(r'$\sqrt{p_x^2 + p_y^2}$')
    axs[1, 1].set_ylabel(ylabel)
    axs[1, 1].legend()


    hist_1, bins_1, _ = axs[2, 0].hist(x_1, bins=bins_x, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[2, 0].hist(x_2, bins=bins_x, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[2, 0].set_title(r'$x$ Position')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel(ylabel)
    axs[2, 0].legend()

    hist_1, bins_1, _ = axs[2, 1].hist(y_1, bins=bins_y, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[2, 1].hist(y_2, bins=bins_y, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[2, 1].set_title(r'$y$ Position')
    axs[2, 1].set_xlabel(r'$y$')
    axs[2, 1].set_ylabel(ylabel)
    axs[2, 1].legend()

    hist_1, bins_1, _ = axs[2, 2].hist(z_1, bins=bins_z, density=density, histtype='step', color='firebrick', label=label_1, log=True)
    hist_2, bins_2, _ = axs[2, 2].hist(z_2, bins=bins_z, density=density, histtype='step', color='steelblue', label=label_2, log=True)
    axs[2, 2].set_title(r'$z$ Position')
    axs[2, 2].set_xlabel(r'$z$')
    axs[2, 2].set_ylabel(ylabel)
    axs[2, 2].legend()

    n = int(1e6)
    axs[1, 2].scatter(x_1[:n], y_1[:n], s=1, alpha=0.1, c=p_mag_1[:n], cmap='Reds', vmin=max(0, args.filter_p[0]), vmax=min(250, args.filter_p[1]), label=label_1)
    axs[1, 2].scatter(x_2[:n], y_2[:n], s=1, alpha=0.1, c=p_mag_2[:n], cmap='Blues', vmin=max(0, args.filter_p[0]), vmax=min(250, args.filter_p[1]), label=label_2)
    axs[1, 2].set_title(r'Y vs X', fontsize=title_fontsize)
    axs[1, 2].set_xlabel(r'$x$ [m]', fontsize=label_fontsize)
    axs[1, 2].set_ylabel(r'$y$ [m]', fontsize=label_fontsize)

    for ax in axs.flatten():
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f'Plot saved to {output_filename}')
    plt.close()

def plot_positions(output_filename, 
                   px_1, py_1, pz_1, x_1, y_1, z_1,
                   px_2, py_2, pz_2, x_2, y_2, z_2,  weights_1=None, weights_2=None,
                   label_1='data1', label_2='data2'):
    """
    Plots histograms of x and y position components side-by-side.
    Saves the plot as a PDF with a transparent background.
    """

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-paper') # A good style for papers
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    density = args.density
    ylabel = 'Density' if density else 'Count'
    num_bins = 100

    # --- Font Sizes for Publication Quality ---
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    # --- Plot X Position ---
    bins_x = np.linspace(min(x_1.min(), x_2.min()), max(x_1.max(), x_2.max()), num_bins)
    axs[0].hist(x_1, bins=bins_x, weights=weights_1, density=density, histtype='step', color='firebrick', label=label_1, lw=1.5, log=True)
    axs[0].hist(x_2, bins=bins_x, weights=weights_2, density=density, histtype='step', color='steelblue', label=label_2, lw=1.5, log=True)
    axs[0].set_title(r'$x$ Position', fontsize=title_fontsize)
    axs[0].set_xlabel(r'$x$ [m]', fontsize=label_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot Y Position ---
    bins_y = np.linspace(min(y_1.min(), y_2.min()), max(y_1.max(), y_2.max()), num_bins)
    axs[1].hist(y_1, bins=bins_y, weights=weights_1, density=density, histtype='step', color='firebrick', label=label_1, lw=1.5, log=True)
    axs[1].hist(y_2, bins=bins_y, weights=weights_2, density=density, histtype='step', color='steelblue', label=label_2, lw=1.5, log=True)
    axs[1].set_title(r'$y$ Position', fontsize=title_fontsize)
    axs[1].set_xlabel(r'$y$ [m]', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # --- Save Figure ---
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', transparent=True)
    print(f'Position plot saved to {output_filename}')
    plt.close()

def plot_momenta(output_filename, 
                 px_1, py_1, pz_1, x_1, y_1, z_1,
                 px_2, py_2, pz_2, x_2, y_2, z_2, 
                 weights_1=None, weights_2=None,
                 label_1='data1', label_2='data2'):
    """
    Plots histograms of px, py, and pz momentum components side-by-side.
    Saves the plot as a PDF with a transparent background.
    """

    
    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    density = args.density
    ylabel = 'Density' if density else 'Count'
    num_bins = 100

    # --- Font Sizes for Publication Quality ---
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    # --- Plot Px ---
    bins_px = np.linspace(min(px_1.min(), px_2.min()), max(px_1.max(), px_2.max()), num_bins)
    axs[0].hist(px_1, bins=bins_px, weights=weights_1, density=density, histtype='step', color='firebrick', label=label_1, lw=1.5, log=True)
    axs[0].hist(px_2, bins=bins_px, weights=weights_2, density=density, histtype='step', color='steelblue', label=label_2, lw=1.5, log=True)
    axs[0].set_title(r'$p_x$ Component', fontsize=title_fontsize)
    axs[0].set_xlabel(r'$p_x$ [GeV]', fontsize=label_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot Py ---
    bins_py = np.linspace(min(py_1.min(), py_2.min()), max(py_1.max(), py_2.max()), num_bins)
    axs[1].hist(py_1, bins=bins_py, weights=weights_1, density=density, histtype='step', color='firebrick', label=label_1, lw=1.5, log=True)
    axs[1].hist(py_2, bins=bins_py, weights=weights_2, density=density, histtype='step', color='steelblue', label=label_2, lw=1.5, log=True)
    axs[1].set_title(r'$p_y$ Component', fontsize=title_fontsize)
    axs[1].set_xlabel(r'$p_y$ [GeV]', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot Pz ---
    bins_pz = np.linspace(min(pz_1.min(), pz_2.min()), max(pz_1.max(), pz_2.max()), num_bins)
    axs[2].hist(pz_1, bins=bins_pz, weights=weights_1, density=density, histtype='step', color='firebrick', label=label_1, lw=1.5, log=True)
    axs[2].hist(pz_2, bins=bins_pz, weights=weights_2, density=density, histtype='step', color='steelblue', label=label_2, lw=1.5, log=True)
    axs[2].set_title(r'$p_z$ Component', fontsize=title_fontsize)
    axs[2].set_xlabel(r'$p_z$ [GeV]', fontsize=label_fontsize)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[2].legend(fontsize=legend_fontsize)
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # --- Save Figure ---
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', transparent=True)
    print(f'Momentum plot saved to {output_filename}')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run muon simulations with specified number of muons.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_filename', type=str, default='histograms_comparison.png', help='Output filename for the histogram plot')
    parser.add_argument('--file_name_1', type=str, help='Path to GEANT4 HDF5 file', default='../data/outputs/results/final_concatenated_results.h5')
    parser.add_argument('--file_name_2', type=str, help='Path to CUDA pickle file', default =  '../data/outputs/outputs_cuda.pkl')
    parser.add_argument('--density', action='store_true', help='Plot histograms with density normalization')
    parser.add_argument('--filter_p', type=float, nargs=2, default=[0.0, 500.0], help='Thresholds to filter muons by min_p <= |P| <= max_p')
    parser.add_argument('--sens_plane', action='store_true', help='Apply sensitive plane cut (|x|<2, |y|<3)')
    parser.add_argument('--propagate_until', type=float, default=None, help='If >0, propagate the loaded data until this z')
    args = parser.parse_args()

    print("Loading data 1...")
    file_name_1 = args.file_name_1
    file_name_2 = args.file_name_2
    # Derive short labels from filenames (basename without extension)
    label_1 = os.path.splitext(os.path.basename(file_name_1))[0]
    label_2 = os.path.splitext(os.path.basename(file_name_2))[0]
    with h5py.File(file_name_1,'r') as f:
        data_1 = {key: f[key][:] for key in f.keys()}
    print("Loaded first data. Shape of data:", len(data_1['px']))

    print("Loading data 2...")
    if file_name_2.endswith('.h5'):
        with h5py.File(file_name_2,'r') as f:
            data_2 = {key: f[key][:] for key in f.keys()}
    else:
        with open(file_name_2, 'rb') as f:
            data_2 = pickle.load(f)
    for key in data_2:
        if not isinstance(data_2[key], np.ndarray): data_2[key] = data_2[key].numpy()
    print("Loaded second data. Shape of data:", len(data_2['px']))


    for key in data_2:
        if key in ['pdg_id', 'W']: continue
        print("=" * 30, key, "=" * 30)
        second_vals = np.array(data_2[key])
        first_vals = np.array(data_1[key])
        print(f"Number of entries - {label_2}: {len(second_vals)}, {label_1}: {len(first_vals)}")
        print(f"{label_2} {key} values: {second_vals}, {label_1} {key} values: {first_vals}")
        print(f"{label_2} {key} average: {second_vals.mean()}, {label_1} {key} average: {first_vals.mean()}")
        print(f"{label_2} {key} std: {second_vals.std()}, {label_1} {key} std: {first_vals.std()}")


    px_1 = data_1['px']
    py_1 = data_1['py']
    pz_1 = data_1['pz']
    x_1 = data_1['x']
    y_1 = data_1['y']
    z_1 = data_1['z']

    px_2 = data_2['px']
    py_2 = data_2['py']
    pz_2 = data_2['pz']
    x_2 = data_2['x']
    y_2 = data_2['y']
    z_2 = data_2['z']

    print("=" * 60)
    print(f'TOTAL number of samples ({label_1}):', px_1.shape[0])
    print(f'TOTAL number of samples ({label_2}):', px_2.shape[0])
    weights_1 = None
    weights_2 = None
    if 'weight' in data_1:
        weights_1 = data_1['weight']
        print(f'TOTAL sum of weights ({label_1}):', np.sum(weights_1))
    if 'weight' in data_2:
        weights_2 = data_2['weight']
        print(f'TOTAL sum of weights ({label_2}):', np.sum(weights_2))

    output_filename = os.path.join('/home/hep/lprate/projects/MuonsAndMatter/plots', args.output_filename)

    px_1, py_1, pz_1, x_1, y_1, z_1, weights_1 = filter_data(
        px_1, py_1, pz_1, x_1, y_1, z_1, weights_1, propagate_until=args.propagate_until)
    
    px_2, py_2, pz_2, x_2, y_2, z_2, weights_2 = filter_data(
        px_2, py_2, pz_2, x_2, y_2, z_2, weights_2, propagate_until=args.propagate_until)

    plot_histograms(output_filename,
                px_1, py_1, pz_1, x_1, y_1, z_1,
                px_2, py_2, pz_2, x_2, y_2, z_2,
                weights_1, weights_2, label_1=label_1, label_2=label_2)
    plot_positions(output_filename.replace('.png','_positions.pdf'),
                px_1, py_1, pz_1, x_1, y_1, z_1,
                px_2, py_2, pz_2, x_2, y_2, z_2,
                weights_1, weights_2, label_1=label_1, label_2=label_2)
    plot_momenta(output_filename.replace('.png','_momenta.pdf'),
                px_1, py_1, pz_1, x_1, y_1, z_1,
                px_2, py_2, pz_2, x_2, y_2, z_2,
                weights_1, weights_2, label_1=label_1, label_2=label_2)

