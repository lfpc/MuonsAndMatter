import time

import numpy as np
import pickle
import torch
import h5py
import os
from utils_cuda_muons.get_geometry import  get_cavern, get_corners_from_detector, create_z_axis_grid, get_magnetic_field
from lib.ship_muon_shield_customfield import get_design_from_params
import lib.reference_designs.params as params_lib

import faster_muons_torch
assert torch.cuda.is_available(), f"CUDA is not available. Torch version: {torch.__version__} \n Torch cuda version: {print(torch.version.cuda)}"

import warnings
# Suppress only the specific torch.storage FutureWarning
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning,
    module="torch.storage"
)


def propagate_muons_with_cuda(
    muons_positions,
    muons_momenta,
    muons_charge,
    arb8_corners,
    cavern_params,
    hist_2d_probability_table_iron,
    hist_2d_alias_table_iron,
    hist_2d_bin_centers_first_dim_iron,
    hist_2d_bin_centers_second_dim_iron,
    hist_2d_bin_widths_first_dim_iron,
    hist_2d_bin_widths_second_dim_iron,
    hist_2d_probability_table_concrete,
    hist_2d_alias_table_concrete,
    hist_2d_bin_centers_first_dim_concrete,
    hist_2d_bin_centers_second_dim_concrete,
    hist_2d_bin_widths_first_dim_concrete,
    hist_2d_bin_widths_second_dim_concrete,
    magnetic_field:dict,
    sensitive_plane_z: float = 82,
    num_steps=100,
    step_length_fixed=0.02,
    seed=1234,
    ):
    # Ensure inputs are float tensors on CUDA
    muons_positions_cuda = muons_positions.float().cuda()
    muons_momenta_cuda = muons_momenta.float().cuda()
    muons_charge = muons_charge.float().cuda()
    hist_2d_probability_table_iron = hist_2d_probability_table_iron.float().cuda()
    hist_2d_alias_table_iron = hist_2d_alias_table_iron.int().cuda()
    hist_2d_bin_centers_first_dim_iron = hist_2d_bin_centers_first_dim_iron.float().cuda()
    hist_2d_bin_centers_second_dim_iron = hist_2d_bin_centers_second_dim_iron.float().cuda()
    hist_2d_bin_widths_first_dim_iron = hist_2d_bin_widths_first_dim_iron.float().cuda()
    hist_2d_bin_widths_second_dim_iron = hist_2d_bin_widths_second_dim_iron.float().cuda()
    hist_2d_probability_table_concrete = hist_2d_probability_table_concrete.float().cuda()
    hist_2d_alias_table_concrete = hist_2d_alias_table_concrete.int().cuda()
    hist_2d_bin_centers_first_dim_concrete = hist_2d_bin_centers_first_dim_concrete.float().cuda()
    hist_2d_bin_centers_second_dim_concrete = hist_2d_bin_centers_second_dim_concrete.float().cuda()
    hist_2d_bin_widths_first_dim_concrete = hist_2d_bin_widths_first_dim_concrete.float().cuda()
    hist_2d_bin_widths_second_dim_concrete = hist_2d_bin_widths_second_dim_concrete.float().cuda()
    cells_arb8, hashed_arb8 = create_z_axis_grid(arb8_corners, sz=15)
    cells_arb8 = cells_arb8.int().contiguous().cuda()
    hashed_arb8 = hashed_arb8.int().contiguous().cuda()
    arb8_corners = arb8_corners.float().cuda()
    cavern_params = cavern_params.float().cuda()

    magnetic_field_B = torch.from_numpy(magnetic_field['B'])
    magnetic_field_ranges = [magnetic_field['range_x'][0], magnetic_field['range_x'][1],
                             magnetic_field['range_y'][0], magnetic_field['range_y'][1],
                             magnetic_field['range_z'][0], magnetic_field['range_z'][1]]
    nx = int(round((magnetic_field_ranges[1] - magnetic_field_ranges[0]) / magnetic_field['range_x'][2])) + 1
    ny = int(round((magnetic_field_ranges[3] - magnetic_field_ranges[2]) / magnetic_field['range_y'][2])) + 1
    nz = int(round((magnetic_field_ranges[5] - magnetic_field_ranges[4]) / magnetic_field['range_z'][2])) + 1
    magnetic_field_B = magnetic_field_B.view(nx, ny, nz, 3).float().cuda().contiguous()
    magnetic_field_ranges = torch.tensor([magnetic_field_ranges]).div(100).float().cpu().contiguous()
    
    kill_at = 0.18
    use_symmetry = True


    t1 = time.time()
    faster_muons_torch.propagate_muons_with_alias_sampling(
        muons_positions_cuda,
        muons_momenta_cuda,
        muons_charge,
        hist_2d_probability_table_iron,
        hist_2d_alias_table_iron,
        hist_2d_bin_centers_first_dim_iron,
        hist_2d_bin_centers_second_dim_iron,
        hist_2d_bin_widths_first_dim_iron,
        hist_2d_bin_widths_second_dim_iron,
        hist_2d_probability_table_concrete,
        hist_2d_alias_table_concrete,
        hist_2d_bin_centers_first_dim_concrete,
        hist_2d_bin_centers_second_dim_concrete,
        hist_2d_bin_widths_first_dim_concrete,
        hist_2d_bin_widths_second_dim_concrete,
        magnetic_field_B,
        magnetic_field_ranges,
        arb8_corners,
        cells_arb8,
        hashed_arb8,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed, 
        seed
    )
    
    torch.cuda.synchronize()
    print("Took", time.time() - t1, "seconds for %.2e muons and %d steps." % (len(muons_positions_cuda), num_steps))

    # Convert results back to numpy arrays and return
    return muons_positions_cuda.cpu(), muons_momenta_cuda.cpu()

def run(params,
        muons:np.array,
        sensitive_plane: dict = {'dz': 0.02, 'dx': 4, 'dy': 6, 'position': 82.0},
        histogram_dir='data',
        save_dir = None,
        n_steps=500,
        SmearBeamRadius=0.0,
        fSC_mag = False,
        field_map_file = None,
        NI_from_B = True,
        use_diluted = False,
        add_cavern = True,
        SND = False,
        return_all = False,
        seed = 0):
    t0 = time.time()
    if seed is None: seed = np.random.randint(0, 10000)
    if not torch.is_tensor(muons): muons = torch.from_numpy(muons).float()
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]
    muons_charge = muons[:,6]
    if muons_charge.abs().eq(13).all(): muons_charge = muons_charge.div(-13)
    assert muons_charge.abs().eq(1).all(), f"PDG IDs or charges in the input file are not correct. They should be either +/-13 or +/-1., {muons_charge.unique(return_counts=True)}" 

    if SmearBeamRadius > 0: #ring transformation
        raise ValueError("Target is not implemented. Do not use SmearBeamRadius > 0")
        sigma = 1.6
        gauss_x = torch.randn(muons_positions.size(0)) * sigma
        gauss_y = torch.randn(muons_positions.size(0)) * sigma
        uniform = torch.rand(muons_positions.size(0))
        _phi = uniform * 2 * np.pi
        dx = SmearBeamRadius * torch.cos(_phi) + gauss_x
        dy = SmearBeamRadius * torch.sin(_phi) + gauss_y
        muons_positions[:,0] = muons_positions[:,0] + (dx / 100)
        muons_positions[:,1] = muons_positions[:,1] + (dy / 100)

    detector = get_design_from_params(params = params,
                      fSC_mag = fSC_mag,
                      simulate_fields=True,
                      sensitive_film_params=None,
                      field_map_file = field_map_file,
                      add_cavern = add_cavern,
                      add_target = False,
                      sensitive_decay_vessel = False,
                      extra_magnet=False,
                      NI_from_B = NI_from_B,
                      use_diluted = use_diluted,
                      SND = SND,
                      cores_field = 8)
    corners = get_corners_from_detector(detector)
    cavern = get_cavern(detector)
    mag_dict = get_magnetic_field(detector)
    print(f"Field + Detector and corners setup took {time.time() - t0:.2f} seconds.")


    with open(os.path.join(histogram_dir, 'alias_histograms_G4_Fe.pkl'), 'rb') as f:
        hist_data = pickle.load(f)
        histograms_iron = [hist_data['hist_2d_probability_table'],hist_data['hist_2d_alias_table'],
                        hist_data['centers_first_dim'], hist_data['centers_second_dim'],
                        hist_data['width_first_dim'], hist_data['width_second_dim']]
        step_length = hist_data['step_length']
    if add_cavern:
        with open(os.path.join(histogram_dir, 'alias_histograms_G4_CONCRETE.pkl'), 'rb') as f:
            hist_data = pickle.load(f)
            histograms_concrete = [hist_data['hist_2d_probability_table'],hist_data['hist_2d_alias_table'],
                            hist_data['centers_first_dim'], hist_data['centers_second_dim'],
                            hist_data['width_first_dim'], hist_data['width_second_dim']]
            assert step_length == hist_data['step_length'], "Step lengths in the two histogram files are different"
    else:
        histograms_concrete = [torch.zeros_like(s) for s in histograms_iron]


    sensitive_plane_z = -2.0 if sensitive_plane is None else sensitive_plane['position']

    print("Using CUDA for propagation... (server)")
    #torch.cuda.synchronize()
    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            muons_charge,
            corners,
            cavern,
            *histograms_iron,
            *histograms_concrete,
            mag_dict,
            sensitive_plane_z - sensitive_plane['dz']/2,
            n_steps,
            step_length,
            seed,
        )
    weights = muons[:,7] if (muons.shape[1]>7) else None
    if sensitive_plane is not None and not return_all:
        in_sens_plane = (out_position[:,0].abs() < sensitive_plane['dx']/2) & \
                        (out_position[:,1].abs() < sensitive_plane['dy']/2) & \
                        (out_position[:,2] >= (sensitive_plane['position'] - sensitive_plane['dz']/2)) #& \
                        #(out_position[:,2] <= (sensitive_plane['position'] + sensitive_plane['dz']/2))

        out_momenta = out_momenta[in_sens_plane]
        out_position = out_position[in_sens_plane]
        muons_charge = muons_charge[in_sens_plane].int()
        print("Number of outputs:", out_momenta.shape[0])
        weights = weights[in_sens_plane] if weights is not None else None

    out_position = out_position.cpu()
    out_momenta = out_momenta.cpu()
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2],
        'pdg_id': muons_charge*(-13)
    }
    if muons.shape[1]>7:
        output['weight'] = weights
        print("Number of HITS (weighted)", weights.sum().item())
    if save_dir is not None:
        t1 = time.time()
        with open(save_dir, 'wb') as f:
            pickle.dump(output, f)
        print("Data saved to", save_dir, "took", time.time() - t1, "seconds to save.")
    return output  





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h',dest = 'histogram_dir', type=str, default='data/',
                        help='Path to the histogram file')
    parser.add_argument('--muons', '-f', dest='input_file', type=str, default="../data/muons/full_sample_after_target.h5",
                        help='Path to input muon file (.npy, .pkl, .h5). If not provided a synthetic example will be used.')
    parser.add_argument('--n_muons', '-n', type=int, default=0,
                        help='Maximum number of muons to load from the input file; 0 means all')
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='Number of steps for simulation')
    parser.add_argument('--sens_plane', type=float, default=82.0,
                        help='Z position of the sensitive plane')
    parser.add_argument("-remove_cavern", dest="add_cavern", action='store_false', help="Remove the cavern from simulation")
    parser.add_argument('-expanded_sens_plane', action='store_true',
                        help='Use extended sensitive plane dimensions')
    parser.add_argument('-plot', action='store_true',
                        help='Plot histograms')
    parser.add_argument("-params", type=str, default='tokanut_v5', help="Magnet parameters configuration - name or file path. Available names: " + ', '.join(params_lib.params.keys()) + ". If 'test', will prompt for input.")
    parser.add_argument('--SmearBeamRadius', type=float, default=0.0,
                        help='Radius of Ring effect applied to input')
    args = parser.parse_args()
    
    if args.params == 'test':
        params_input = input("Enter the params as a Python list (e.g., [1.0, 2.0, 3.0]): ")
        params = eval(params_input)
    elif args.params in params_lib.params.keys():
        params = params_lib.params[args.params]
    elif os.path.isfile(args.params):
        with open(args.params, "r") as txt_file:
            params = [float(line.strip()) for line in txt_file]
    else: 
        raise ValueError(f"Invalid params: {args.params}. Must be a valid parameter name or a file path. \
                         Avaliable names: {', '.join(params_lib.params.keys())}.")
    params = np.asarray(params)
    time0 = time.time()
    if args.input_file is not None:
        input_file = args.input_file
        print(f"Loading muons from {input_file}")
        if input_file.endswith('.npy') or input_file.endswith('.pkl'):
            muons_iter = np.load(input_file, allow_pickle=True, mmap_mode='r')
            muons = muons_iter[:args.n_muons].copy() if args.n_muons > 0 else muons_iter[:].copy()
        elif input_file.endswith('.h5'):
            with h5py.File(input_file, 'r') as f:
                px = f['px'][: args.n_muons] if args.n_muons > 0 else f['px'][:]
                py = f['py'][: args.n_muons] if args.n_muons > 0 else f['py'][:]
                pz = f['pz'][: args.n_muons] if args.n_muons > 0 else f['pz'][:]
                x = f['x'][: args.n_muons] if args.n_muons > 0 else f['x'][:]
                y = f['y'][: args.n_muons] if args.n_muons > 0 else f['y'][:]
                z = f['z'][: args.n_muons] if args.n_muons > 0 else f['z'][:]
                pdg = f['pdg'][: args.n_muons] if args.n_muons > 0 else f['pdg'][:]
                weight = f['weight'][: args.n_muons] if args.n_muons > 0 else f['weight'][:]
                muons = np.stack([px, py, pz, x, y, z, pdg, weight], axis=1).astype(np.float64)
    print(f"Loaded {muons.shape[0]} muons. Took {time.time() - time0:.2f} seconds to load.")
    if args.expanded_sens_plane and args.add_cavern: dx,dy = 9.,6. 
    elif args.expanded_sens_plane: dx,dy = 9.,6.#14,14
    else: dx, dy = 4.0, 6.0
    sensitive_film_params = {'dz': 0.01, 'dx': dx, 'dy': dy, 'position':args.sens_plane} if args.sens_plane >0 else None
    t_run_start = time.time()
    output = run(params, muons, sensitive_film_params, 
                 histogram_dir=args.histogram_dir, n_steps=args.n_steps, 
                 SmearBeamRadius=args.SmearBeamRadius, fSC_mag=False, NI_from_B=True, use_diluted=False, add_cavern=args.add_cavern,
                 field_map_file= None,
                 save_dir="../data/outputs/outputs_cuda.pkl")
    print(f"Run completed in {time.time() - t_run_start:.2f} seconds.")
    if args.plot:
        import matplotlib.pyplot as plt
        out_dir = "plots/outputs"

        os.makedirs(out_dir, exist_ok=True)
        input_data = {
            'px': muons[:,0],
            'py': muons[:,1],
            'pz': muons[:,2],
            'x': muons[:,3],
            'y': muons[:,4],
            'z': muons[:,5],
            'pdg_id': muons[:,6],
            'weight': muons[:,7]
        }
        print('Number of INPUTS:', muons.shape[0], ' (weighted: ', muons[:,7].sum().item())
        print('Number of OUTPUTS:', output['x'].shape[0], ' (weighted: ', output['weight'].sum().item())
        for key, values in output.items():
            plt.figure()
            plt.hist(input_data[key], bins='auto', histtype='step', label='Input', linewidth=1.5, log=True)
            plt.hist(values, bins='auto', histtype='step', label='Output', linewidth=1.5, log=True)
            plt.title(f"Histogram of {key} (CUDA_MUONS)")
            plt.xlabel(key)
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{key}_hist.png")
            plt.close()
        print(f"Histograms saved in {out_dir}")
