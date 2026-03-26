import time
import numpy as np
import pickle
import torch
import h5py
import os
from lib.ship_muon_shield import get_design_from_params
import lib.reference_designs.params as params_lib
from cuda_muons import set_environment, propagate_muons_with_cuda

assert torch.cuda.is_available(), f"CUDA is not available. Torch version: {torch.__version__} \n Torch cuda version: {print(torch.version.cuda)}"




def get_corners_from_detector(detector, use_symmetry=True):
    def expand_corners(corners, dz, z_center):
        corners = np.array(corners).reshape(8, 2)
        z = np.full((8, 1), z_center - dz)
        z[4:] = z_center + dz
        return np.hstack([corners, z])

    all_corners = []
    for magnet in detector['magnets']:
        components = magnet['components'][:3] if use_symmetry else magnet['components']
        for comp in components:
            all_corners.append(expand_corners(comp['corners'], comp['dz'], comp['z_center']))
    return torch.from_numpy(np.array(all_corners))


def get_cavern_from_detector(detector):
    if 'cavern' not in detector:
        return torch.tensor([[-30, 30, -30, 30, 0],
                             [-30, 30, -30, 30, 0]], dtype=torch.float32)
    TCC8, ECN3 = detector['cavern'][0], detector['cavern'][1]
    return torch.tensor([
        [TCC8['x1'], TCC8['x2'], TCC8['y1'], TCC8['y2'], TCC8['z_center'] + TCC8['dz']],
        [ECN3['x1'], ECN3['x2'], ECN3['y1'], ECN3['y2'], ECN3['z_center'] - ECN3['dz']],
    ], dtype=torch.float32)


def get_magnetic_field_from_detector(detector, use_symmetry=True):
    """Extract magnetic field from detector dict.

    Returns either a field map dict (for FEM-simulated fields) or a tensor
    of shape (N_arb8s, 3) with uniform [Bx, By, Bz] per ARB8.
    """
    if 'global_field_map' in detector and detector['global_field_map']:
        # Field map mode
        mag_dict = detector['global_field_map']
        if isinstance(mag_dict['B'], str):
            with h5py.File(mag_dict['B'], 'r') as f:
                mag_dict['B'] = f['B'][:]
        return mag_dict
    else:
        # Uniform field mode: extract field vectors from each component
        all_fields = []
        for magnet in detector['magnets']:
            components = magnet['components'][:3] if use_symmetry else magnet['components']
            for comp in components:
                all_fields.append(comp['field'])
        return torch.tensor(all_fields, dtype=torch.float32)


def load_histogram(filepath):
    with open(filepath, 'rb') as f:
        hist_data = pickle.load(f)
    return [
        hist_data['hist_2d_probability_table'],
        hist_data['hist_2d_alias_table'],
        hist_data['centers_first_dim'],
        hist_data['centers_second_dim'],
        hist_data['width_first_dim'],
        hist_data['width_second_dim'],
    ], hist_data['step_length']



def run_from_params(
    params,
    muons: np.ndarray,
    sensitive_plane={'dz': 0.02, 'dx': 4, 'dy': 6, 'position': 82.0},
    histogram_dir='data',
    save_dir=None,
    n_steps=500,
    fSC_mag=False,
    field_map_file=None,
    NI_from_B=True,
    use_diluted=False,
    add_cavern=True,
    simulate_fields=False,
    SND=False,
    cores_field=8,
    return_all=False,
    SmearBeamRadius=0.0,
    seed=0,
    device='cuda',
):
    """Run muon propagation from magnet parameters (detector-based geometry).

    Args:
        params: Array of shape (N_magnets, 15) with magnet parameters.
        muons: Input muon array [px, py, pz, x, y, z, pdg_id, weight].
        sensitive_plane: Dict with 'dz','dx','dy','position', or list of such dicts.
        histogram_dir: Directory containing alias histogram files.
        save_dir: If provided, save output to this path.
        n_steps: Number of propagation steps.
        fSC_mag: Whether superconducting magnets are used.
        field_map_file: Path to field map file (only used if simulate_fields=True).
        NI_from_B: Whether NI is derived from B (affects SC threshold).
        use_diluted: Whether to use diluted steel in FEM simulation.
        add_cavern: Whether to include cavern geometry.
        simulate_fields: If True, use FEM-simulated field map (slow).
        SND: Use SND detector geometry.
        cores_field: Number of CPU cores for field simulation.
        return_all: If True, return all muons; if False, filter by sensitive plane.
        seed: Random seed for propagation.
        device: Device to run on ('cuda' or GPU index).

    Returns:
        Dict with 'px','py','pz','x','y','z','pdg_id', and optionally 'weight'.
    """
    t0 = time.time()
    if seed is None:
        seed = np.random.randint(0, 10000)
    if not torch.is_tensor(muons):
        muons = torch.from_numpy(muons).float()

    # === SETUP (done once, regardless of number of sensitive planes) ===
    use_symmetry = True
    assert SmearBeamRadius == 0.0, "Smearing not implemented in this version. Please set SmearBeamRadius=0.0"

    detector = get_design_from_params(
        params=params,
        fSC_mag=fSC_mag,
        simulate_fields=simulate_fields,
        sensitive_film_params=None,
        field_map_file=field_map_file,
        add_cavern=add_cavern,
        add_target=False,
        sensitive_decay_vessel=False,
        extra_magnet=False,
        NI_from_B=NI_from_B,
        use_diluted=use_diluted,
        SND=SND,
        cores_field=cores_field,
    )
    corners = get_corners_from_detector(detector, use_symmetry=use_symmetry)
    cavern = get_cavern_from_detector(detector)
    mag_field = get_magnetic_field_from_detector(detector, use_symmetry=use_symmetry)
    print(f"Detector + field setup took {time.time() - t0:.2f} seconds.")

    # Load material histograms
    iron_hists, step_length = load_histogram(os.path.join(histogram_dir, 'alias_histograms_G4_Fe.pkl'))
    material_histograms = {'iron': iron_hists}
    if add_cavern:
        concrete_hists, concrete_step = load_histogram(os.path.join(histogram_dir, 'alias_histograms_G4_CONCRETE.pkl'))
        assert step_length == concrete_step, "Step lengths in the two histogram files are different"
        material_histograms['concrete'] = concrete_hists
    else:
        material_histograms['concrete'] = [torch.zeros_like(h) for h in iron_hists]

    # Pre-compute and transfer static data to GPU (done once)
    environment = set_environment(
        corners, cavern, material_histograms, mag_field, device=device,
    )

    # === Prepare muons ===
    muons_momenta = muons[:, :3]
    muons_positions = muons[:, 3:6]
    muons_charge = muons[:, 6]
    weights = muons[:, 7] if muons.shape[1] > 7 else None
    if muons_charge.abs().eq(13).all():
        muons_charge = muons_charge.div(-13)
    assert muons_charge.abs().eq(1).all(), \
        f"PDG IDs / charges must be +/-13 or +/-1. Got: {muons_charge.unique(return_counts=True)}"

    # === PROPAGATION (per sensitive plane) ===
    planes = sensitive_plane if isinstance(sensitive_plane, list) else [sensitive_plane]

    for i, plane in enumerate(planes):
        if muons_positions.shape[0] == 0:
            print("No muons left to propagate.")
            break

        sens_z = n_steps * step_length if plane is None else plane['position'] - plane['dz'] / 2
        print(f"Using CUDA for propagation... (plane {i}, z={sens_z:.2f})")

        out_position, out_momenta = propagate_muons_with_cuda(
            muons_positions,
            muons_momenta,
            muons_charge,
            environment,
            sens_z,
            n_steps,
            step_length,
            use_symmetry,
            seed,
            device,
        )

        if plane is not None:
            in_sens_plane = (
                (out_position[:, 0].abs() < plane['dx'] / 2)
                & (out_position[:, 1].abs() < plane['dy'] / 2)
                & (out_position[:, 2] >= (plane['position'] - plane['dz'] / 2))
            )
            print(f"Plane {i} (Z={plane['position']}): {in_sens_plane.sum().item()} muons remaining.")
            if return_all:
                out_momenta[~in_sens_plane] = 0.0
            else:
                out_momenta = out_momenta[in_sens_plane]
                out_position = out_position[in_sens_plane]
                muons_charge = muons_charge[in_sens_plane]
                if weights is not None:
                    weights = weights[in_sens_plane]

        # Feed outputs as inputs for the next plane
        muons_positions = out_position
        muons_momenta = out_momenta

    # === Build output dict ===
    out_position = out_position.cpu()
    out_momenta = out_momenta.cpu()
    output = {
        'px': out_momenta[:, 0],
        'py': out_momenta[:, 1],
        'pz': out_momenta[:, 2],
        'x': out_position[:, 0],
        'y': out_position[:, 1],
        'z': out_position[:, 2],
        'pdg_id': muons_charge.cpu() * (-13),
    }
    if weights is not None:
        output['weight'] = weights.cpu()
        print("Number of HITS (weighted)", weights.sum().item())

    if save_dir is not None:
        t1 = time.time()
        with open(save_dir, 'wb') as f:
            pickle.dump(output, f)
        print(f"Data saved to {save_dir}, took {time.time() - t1:.2f} seconds.")
    return output



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h', dest='histogram_dir', type=str, default='cuda_muons/data/',
                        help='Path to the histogram file')
    parser.add_argument('-muons', '--f', dest='input_file', type=str,
                        default='data/muons/full_sample_after_target.h5',
                        help='Path to input muon file (.npy, .pkl, .h5).')
    parser.add_argument('-n_muons', '--n', dest='n_muons', type=int, default=0,
                        help='Maximum number of muons to load (0 = all)')
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='Number of steps for simulation')
    parser.add_argument('-sens_plane', type=float, default=82.0,
                        help='Z position of the sensitive plane')
    parser.add_argument('-uniform_fields', dest='simulate_fields', action='store_false',
                        help='Use uniform fields instead of realistic field maps (FEM)')
    parser.add_argument('-remove_cavern', dest='add_cavern', action='store_false',
                        help='Remove the cavern from simulation')
    parser.add_argument('-expanded_sens_plane', action='store_true',
                        help='Use extended sensitive plane dimensions')
    parser.add_argument('-plot', action='store_true', help='Plot histograms')
    parser.add_argument('-SND', action='store_true', help='Use SND detector geometry')
    parser.add_argument('-diluted_iron', action='store_true', help='Use diluted field map')
    parser.add_argument('-params', type=str, default='tokanut_v5',
                        help='Magnet parameters: name or file path. '
                        f"Available: {', '.join(params_lib.params.keys())}. 'test' = manual input.")
    parser.add_argument('--gpu', dest='gpu', type=int, default=0,
                        help='GPU index to use')
    args = parser.parse_args()

    # Load params
    if args.params == 'test':
        params = eval(input("Enter the params as a Python list (e.g., [1.0, 2.0, 3.0]): "))
    elif args.params in params_lib.params:
        params = params_lib.params[args.params]
    elif os.path.isfile(args.params):
        with open(args.params, 'r') as f:
            params = [float(line.strip()) for line in f]
    else:
        raise ValueError(f"Invalid params: {args.params}. Must be a name or file path. "
                         f"Available: {', '.join(params_lib.params.keys())}.")
    params = np.asarray(params).reshape(-1, 15)

    # Load muons
    time0 = time.time()
    input_file = args.input_file
    print(f"Loading muons from {input_file}")
    if input_file.endswith('.npy') or input_file.endswith('.pkl'):
        muons_iter = np.load(input_file, allow_pickle=True, mmap_mode='r')
        muons = muons_iter[:args.n_muons].copy() if args.n_muons > 0 else muons_iter[:].copy()
    elif input_file.endswith('.h5'):
        with h5py.File(input_file, 'r') as f:
            n = args.n_muons if args.n_muons > 0 else None
            muons = np.stack([f[k][:n] for k in ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight')],
                             axis=1).astype(np.float64)
    print(f"Loaded {muons.shape[0]} muons. Took {time.time() - time0:.2f} seconds.")

    # Sensitive plane
    dx, dy = (9.0, 6.0) if args.expanded_sens_plane else (4.0, 6.0)
    sensitive_film_params = {'dz': 0.01, 'dx': dx, 'dy': dy, 'position': args.sens_plane} \
        if args.sens_plane > 0 else None

    # Run
    t_run_start = time.time()
    output = run_from_params(
        params, muons, sensitive_film_params,
        histogram_dir=args.histogram_dir,
        n_steps=args.n_steps,
        fSC_mag=False,
        simulate_fields=args.simulate_fields,
        NI_from_B=True,
        use_diluted=args.diluted_iron,
        add_cavern=args.add_cavern,
        field_map_file=None,
        SND=args.SND,
        save_dir='data/outputs/outputs_cuda.pkl',
        device=args.gpu,
    )
    print(f"Run completed in {time.time() - t_run_start:.2f} seconds.")

    if args.plot:
        import matplotlib.pyplot as plt
        out_dir = 'plots/outputs'
        os.makedirs(out_dir, exist_ok=True)
        input_data = {k: muons[:, i] for i, k in enumerate(
            ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg_id', 'weight'])}
        print(f"Number of INPUTS: {muons.shape[0]}  (weighted: {muons[:,7].sum().item():.1f})")
        print(f"Number of OUTPUTS: {output['x'].shape[0]}  (weighted: {output['weight'].sum().item():.1f})")
        for key, values in output.items():
            plt.figure()
            plt.hist(input_data[key], bins='auto', histtype='step', label='Input', linewidth=1.5, log=True)
            plt.hist(values, bins='auto', histtype='step', label='Output', linewidth=1.5, log=True)
            plt.title(f"Histogram of {key} (CUDA_MUONS)")
            plt.xlabel(key)
            plt.ylabel('Counts')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{out_dir}/{key}_hist.png')
            plt.close()
        print(f"Histograms saved in {out_dir}")
