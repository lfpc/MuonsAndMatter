import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import os


def hist_with_errors(ax, data, bins=10, density=False, histtype='step', errors=True, **kwargs):
    """
    Wrapper function to plot histogram with optional error bars.

    Parameters:
        ax : matplotlib.axes.Axes
            Axis on which to plot the histogram.
        data : array-like
            Data to plot in the histogram.
        bins : int or sequence, optional
            Number of bins or bin edges.
        density : bool, optional
            If True, normalize the histogram.
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional
            The type of histogram to draw.
        errors : bool, optional
            If True, add error bars to the histogram.
        **kwargs : additional keyword arguments
            Additional arguments passed to the hist function.

    Returns:
        n : array or list of arrays
            The values of the histogram bins.
        bins : array
            The edges of the bins.
        patches : list or list of lists
            Silent list of individual patches used to create the histogram.
    """
    # Calculate the histogram
    n, bins, patches = ax.hist(data, bins=bins, density=density, histtype=histtype, **kwargs)

    if errors:
        # Calculate the bin centers
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Calculate error bars assuming Poisson (sqrt(n) for each bin count)
        if density:
            # For density, scale errors accordingly
            bin_widths = np.diff(bins)
            errors = np.sqrt(n) / (len(data) * bin_widths)  # Error with density normalization
        else:
            errors = np.sqrt(n)  # Error without density normalization

        # Plot error bars
        ax.errorbar(bin_centers, n, yerr=errors, fmt='none', ecolor='black', capsize=3, label='Error bars')

    return n, bins, patches



def plot_histograms(data_geant4, data_cuda, geant4_filter, cuda_filter, output_filename):
    # Apply Geant4 filter
    px = data_geant4['px'][geant4_filter]
    py = data_geant4['py'][geant4_filter]
    pz = data_geant4['pz'][geant4_filter]
    x = data_geant4['x'][geant4_filter]
    y = data_geant4['y'][geant4_filter]
    z = data_geant4['z'][geant4_filter]

    # Apply CUDA filter
    px_cuda = data_cuda['px'][cuda_filter]
    py_cuda = data_cuda['py'][cuda_filter]
    pz_cuda = data_cuda['pz'][cuda_filter]
    x_cuda = data_cuda['x'][cuda_filter]
    y_cuda = data_cuda['y'][cuda_filter]
    z_cuda = data_cuda['z'][cuda_filter]

    print("Length of G4 is",int(np.sum(geant4_filter)), "and length of CUDA is %d."%int(np.sum(cuda_filter)))

    # Calculate total distance and momentum magnitudes
    dist_total = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dist_total_cuda = np.sqrt(x_cuda ** 2 + y_cuda ** 2 + z_cuda ** 2)

    p_mag = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    p_mag_cuda = np.sqrt(px_cuda ** 2 + py_cuda ** 2 + pz_cuda ** 2)

    # Calculate transverse momentum sqrt(px^2 + py^2) for both datasets
    p_transverse = np.sqrt(px ** 2 + py ** 2)
    p_transverse_cuda = np.sqrt(px_cuda ** 2 + py_cuda ** 2)


    # Define the number of bins and compute bins for each metric based on Geant4 data range
    num_bins = 50
    bins_px = np.linspace(px_cuda.min(), px_cuda.max(), num_bins)
    bins_py = np.linspace(py_cuda.min(), py_cuda.max(), num_bins)
    bins_pz = np.linspace(pz_cuda.min(), pz_cuda.max(), num_bins)
    bins_p_mag = np.linspace(p_mag_cuda.min(), p_mag_cuda.max(), num_bins)
    bins_p_transverse = np.linspace(p_transverse.min(), p_transverse.max(), num_bins)
    bins_dist_total = np.linspace(dist_total.min(), dist_total.max(), num_bins)

    # print("Check A", np.sum(px_cuda>15), np.sum(p_transverse_cuda>15))
    # print(bins_p_transverse)
    # 0/0


    bins_x = np.linspace(x_cuda.min(), x_cuda.max(), num_bins)
    bins_y = np.linspace(y_cuda.min(), y_cuda.max(), num_bins)
    bins_z = np.linspace(z.min(), z.max(), num_bins)

    # Plotting with 3x3 layout
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: Plot px, py, pz comparison
    axs[0, 0].hist(px, bins=bins_px, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[0, 0].hist(px_cuda, bins=bins_px, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 0].set_title(r'$p_x$ Component')
    axs[0, 0].set_xlabel(r'$p_x$')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()


    axs[0, 1].hist(py, bins=bins_py, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[0, 1].hist(py_cuda, bins=bins_py, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 1].set_title(r'$p_y$ Component')
    axs[0, 1].set_xlabel(r'$p_y$')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()

    axs[0, 2].hist(pz, bins=bins_pz, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[0, 2].hist(pz_cuda, bins=bins_pz, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[0, 2].set_title(r'$p_z$ Component')
    axs[0, 2].set_xlabel(r'$p_z$')
    axs[0, 2].set_ylabel('Density')
    axs[0, 2].legend()

    # Row 2: Plot p_mag, p_transverse, dist_total comparison
    axs[1, 0].hist(p_mag, bins=bins_p_mag, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[1, 0].hist(p_mag_cuda, bins=bins_p_mag, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[1, 0].set_title(r'Momentum Magnitude $(|p|)$')
    axs[1, 0].set_xlabel(r'$|p|$')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()

    axs[1, 1].hist(p_transverse, bins=bins_p_transverse, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[1, 1].hist(p_transverse_cuda, bins=bins_p_transverse, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[1, 1].set_title(r'Transverse Momentum $(\sqrt{p_x^2 + p_y^2})$')
    axs[1, 1].set_xlabel(r'$\sqrt{p_x^2 + p_y^2}$')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].legend()

    axs[1, 2].hist(dist_total, bins=bins_dist_total, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[1, 2].hist(dist_total_cuda, bins=bins_dist_total, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[1, 2].set_title(r'Total Distance $(\sqrt{x^2 + y^2 + z^2})$')
    axs[1, 2].set_xlabel(r'$\sqrt{x^2 + y^2 + z^2}$')
    axs[1, 2].set_ylabel('Density')
    axs[1, 2].legend()

    # Row 3: Plot x, y, z comparison
    axs[2, 0].hist(x, bins=bins_x, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[2, 0].hist(x_cuda, bins=bins_x, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 0].set_title(r'$x$ Position')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel('Density')
    axs[2, 0].legend()

    axs[2, 1].hist(y, bins=bins_y, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[2, 1].hist(y_cuda, bins=bins_y, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 1].set_title(r'$y$ Position')
    axs[2, 1].set_xlabel(r'$y$')
    axs[2, 1].set_ylabel('Density')
    axs[2, 1].legend()

    axs[2, 2].hist(z, bins=bins_z, density=True, histtype='step', color='firebrick', label='Geant4', log=True)
    axs[2, 2].hist(z_cuda, bins=bins_z, density=True, histtype='step', color='steelblue', label='CUDA', log=True)
    axs[2, 2].set_title(r'$z$ Position')
    axs[2, 2].set_xlabel(r'$z$')
    axs[2, 2].set_ylabel('Density')
    axs[2, 2].legend()

    for ax in axs.flatten():
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

    


# Generate plots
#plot_histograms(data_geant4, data_cuda, np.ones_like(data_geant4['x'], dtype=bool), np.ones_like(data_cuda['x'], dtype=bool), 'plots/hists_a/g4vscuda.pdf')
#plot_histograms(data_geant4, data_cuda, geant4_filter, cuda_filter, 'plots/hists_a/g4vscuda_filt_dist.pdf')

if __name__ == "__main__":

    #with open("data/histograms.pkl", "rb") as file:
    #    hists_dict = pickle.load(file)[(190,200)]
    

    # the_gfile = 'data/data_batch_1/joined.pklj'
    the_gfile = 'data/outputs_geant4.pkl'
    with open(the_gfile, 'rb') as f:
        data_geant4 = pickle.load(f)
    print("Loaded G4 data. Shape of data:", len(data_geant4['px']))

    print("Loading CUDA data...")
    with open('data/outputs_cuda.pkl', 'rb') as f:
        data_cuda = pickle.load(f)
    print("Loaded CUDA data. Shape of data:", len(data_cuda['px']))

    # Plot histograms for each key in data_cuda and data_geant4 (last value for each list)

    output_dir = "plots/hists_comparisons"
    os.makedirs(output_dir, exist_ok=True)

    for key in data_cuda:
        print("=" * 30, key, "=" * 30)
        cuda_vals = np.array(data_cuda[key])
        geant4_vals = np.array(data_geant4[key])
        print(f"CUDA {key} values: {cuda_vals}, Geant4 {key} values: {geant4_vals}")
        print(f"CUDA {key} average: {cuda_vals.mean()}, Geant4 {key} average: {geant4_vals.mean()}")
        print(f"CUDA {key} std: {cuda_vals.std()}, Geant4 {key} std: {geant4_vals.std()}")


        plt.figure(figsize=(8, 5))
        if key == 'pz': bins = np.linspace(100, 195, 100)
        else: bins = 100
        plt.hist(geant4_vals, bins=bins, density=True, histtype='step', color='firebrick', label='Geant4 (last step)')
        plt.hist(cuda_vals, bins=bins, density=True, histtype='step', color='steelblue', label='CUDA')
        plt.xlabel(key)
        plt.ylabel("Density")
        if key == 'pz': plt.xlim(bins[0], bins[-1])
        plt.title(f"Histogram of {key}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{key}_hist.png")
        plt.close()
        print("Saved histogram for", key)

    bins = np.linspace(-10, 0, 100)
    if False:
        import h5py
        with h5py.File("/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/data/muon_data_energy_loss_sens_step_2.h5", "r") as f:
            energy_seg = '(195, 200)'
            pz = f[energy_seg]['pz'][:]
            p0 = f[energy_seg]['initial_momenta'][:]
        delta_hist = (pz - p0) / p0
        plt.hist(np.log(-delta_hist), bins=bins, density=False, histtype='step', color='green', label='Sampling Geant4', log=True, linewidth=2)

    delta = (data_geant4['pz'] - 195) / 195
    delta_cuda = (data_cuda['pz'] - 195) / 195

    plt.hist(np.log(-delta + 1e-10), bins=bins, density=False, histtype='step', color='firebrick', label='Geant4', log=True)
    plt.hist(np.log(-delta_cuda + 1e-10), bins=bins, density=False, histtype='step', color='steelblue', label='CUDA', log=True)
    plt.xlabel(r'$\Delta p_z / p_z$')
    plt.ylabel('Density')
    plt.title(r'$\Delta p_z / p_z$ Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/delta_pz_hist.png")
    plt.close()
    print("Saved histogram for delta pz")



