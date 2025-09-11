import time

import numpy as np
import pickle
import gzip
import torch
import h5py
import os
assert torch.cuda.is_available(), f"CUDA is not available. Torch version: {torch.__version__} \n Torch cuda version: {print(torch.version.cuda)}"
from get_geometry import  get_cavern, get_corners_from_detector
from lib.ship_muon_shield_customfield import get_design_from_params
import lib.reference_designs.params as params_lib

import faster_muons_torch

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
    arb8_corners = arb8_corners.float().cuda()
    cavern_params = cavern_params.float().cuda()

    magnetic_field_B = torch.from_numpy(magnetic_field['B']).float().cuda()
    magnetic_field_ranges = [magnetic_field['range_x'][0], magnetic_field['range_x'][1],
                             magnetic_field['range_y'][0], magnetic_field['range_y'][1],
                             magnetic_field['range_z'][0], magnetic_field['range_z'][1]]
    nx = int(round((magnetic_field_ranges[1] - magnetic_field_ranges[0]) / magnetic_field['range_x'][2])) + 1
    ny = int(round((magnetic_field_ranges[3] - magnetic_field_ranges[2]) / magnetic_field['range_y'][2])) + 1
    nz = int(round((magnetic_field_ranges[5] - magnetic_field_ranges[4]) / magnetic_field['range_z'][2])) + 1
    magnetic_field_B = magnetic_field_B.view(nx, ny, nz, 3).contiguous()
    magnetic_field_ranges = torch.tensor([magnetic_field_ranges]).div(100).float().cpu().contiguous()

    t1 = time.time()
    kill_at = 1.0

    gx, gy, gz = 10, 10, 10
    arb8_grid_range_np = magnetic_field_ranges  # [minX,maxX,minY,maxY,minZ,maxZ]
    total_cells = gx * gy * gz
    arb8_grid_index = torch.zeros(total_cells + 1, dtype=torch.int32, device = 'cuda')  # start offsets per cell
    arb8_grid_list = torch.zeros(0, dtype=torch.int32)                # flat list of Arb8 indices

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
        arb8_grid_index,
        arb8_grid_range_np.float(),
        cavern_params,
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
        histogram_dir='data/alias_histograms.pkl',
        save_dir = None,
        n_steps=500,
        SmearBeamRadius=5.0,
        fSC_mag = False,
        field_map_file = None,
        NI_from_B = True,
        use_diluted = False,
        add_cavern = True,
        SND = False):
    muons = torch.from_numpy(muons).float()
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]
    muons_charge = muons[:,6]

    if SmearBeamRadius > 0: #ring transformation
        sigma = 1.6
        gauss_x = torch.randn(muons_positions.size(0)) * sigma
        gauss_y = torch.randn(muons_positions.size(0)) * sigma
        uniform = torch.rand(muons_positions.size(0))
        _phi = uniform * 2 * np.pi
        dx = SmearBeamRadius * torch.cos(_phi) + gauss_x
        dy = SmearBeamRadius * torch.sin(_phi) + gauss_y
        muons_positions[:,0] += dx / 100
        muons_positions[:,1] += dy / 100

    t0 = time.time()
    detector = get_design_from_params(params = params,
                      force_remove_magnetic_field= False,
                      fSC_mag = fSC_mag,
                      simulate_fields=False,
                      sensitive_film_params=None,
                      field_map_file = field_map_file,
                      add_cavern = True,
                      add_target = False,
                      sensitive_decay_vessel = False,
                      extra_magnet=False,
                      NI_from_B = NI_from_B,
                      use_diluted = use_diluted,
                      SND = SND,
                      cores_field = 8)
    corners = get_corners_from_detector(detector)
    cavern = get_cavern(detector)
    mag_dict = detector['global_field_map']
    with h5py.File(os.path.join("/home/hep/lprate/projects/MuonsAndMatter", mag_dict['B']), 'r') as f:
        mag_dict['B'] = f["B"][:]
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
        histograms_concrete = [torch.zeros(1), torch.zeros(1, dtype=torch.int32),
                            torch.zeros(1), torch.zeros(1),
                            torch.zeros(1), torch.zeros(1)]


    sensitive_plane_z = -2.0 if sensitive_plane is None else sensitive_plane['position']

    print("Using CUDA for propagation... (server)")
    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            muons_charge,
            corners,
            cavern,
            *histograms_iron,
            *histograms_concrete,
            mag_dict,
            sensitive_plane_z,
            n_steps,
            step_length,
            200,
        )
    if sensitive_plane is not None:
        in_sens_plane = (out_position[:,0].abs() < sensitive_plane['dx']/2) & \
                        (out_position[:,1].abs() < sensitive_plane['dy']/2) & \
                        (out_position[:,2] >= sensitive_plane['position'] - sensitive_plane['dz']/2)
        out_momenta = out_momenta[in_sens_plane]
        out_position = out_position[in_sens_plane]
        print("Number of HITS:", out_momenta.shape[0])
    out_momenta = out_momenta.numpy()
    out_position = out_position.numpy()
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2],
        'pdg_id': np.ones_like(out_position[:,0])*-13
    }
    if save_dir is not None:
        with gzip.open(save_dir, 'wb') as f:
            pickle.dump(output, f)
        print("Data saved to", save_dir)
    return output  





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h',dest = 'histogram_dir', type=str, default='data/',
                        help='Path to the histogram file')
    parser.add_argument('--muons', '-f', dest='input_file', type=str, default="/home/hep/lprate/projects/MuonsAndMatter/data/muons/subsample_1M.npy",
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
    parser.add_argument("-params", type=str, default='tokanut_v5', help="Magnet parameters configuration - name or file path. Available names: " + ', '.join(params_lib.params.keys()) + ". If 'test', will prompt for input.")
    
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

    if args.input_file is not None:
        input_file = args.input_file
        print(f"Loading muons from {input_file}")
        if input_file.endswith('.npy') or input_file.endswith('.pkl'):
            muons = np.load(input_file, allow_pickle=True).astype(np.float32)
            if args.n_muons > 0:
                muons = muons[: args.n_muons]
            muons[:,6] = -1*np.sign(muons[:,6])
        elif input_file.endswith('.h5'):
            with h5py.File(input_file, 'r') as f:
                px = f['px'][: args.n_muons] if args.n_muons > 0 else f['px'][:]
                py = f['py'][: args.n_muons] if args.n_muons > 0 else f['py'][:]
                pz = f['pz'][: args.n_muons] if args.n_muons > 0 else f['pz'][:]
                x = f['x'][: args.n_muons] if args.n_muons > 0 else f['x'][:]
                y = f['y'][: args.n_muons] if args.n_muons > 0 else f['y'][:]
                z = f['z'][: args.n_muons] if args.n_muons > 0 else f['z'][:]
                charge = f['pdg_id'][: args.n_muons] if args.n_muons > 0 else f['charge'][:]
                charge = -np.sign(charge)
                muons = np.stack([px, py, pz, x, y, z, charge], axis=1).astype(np.float32)
    if args.expanded_sens_plane and args.add_cavern: dx,dy = 9.,7. 
    elif args.expanded_sens_plane: dx,dy = 14,14
    else: dx, dy = 4.0, 6.0
    sensitive_film_params = {'dz': 0.01, 'dx': dx, 'dy': dy, 'position':args.sens_plane} if args.sens_plane >0 else None

    output = run(params, muons.copy(), sensitive_film_params, 
                 histogram_dir=args.histogram_dir, n_steps=args.n_steps, 
                    SmearBeamRadius=5.0, fSC_mag=False, NI_from_B=True, use_diluted=False,add_cavern = args.add_cavern,
                 field_map_file='/home/hep/lprate/projects/MuonsAndMatter/data/outputs/fields_mm.h5',
                 save_dir="/home/hep/lprate/projects/MuonsAndMatter/cuda_muons/data/outputs_cuda.pkl")

    import matplotlib.pyplot as plt
    out_dir = "plots/outputs"

    os.makedirs(out_dir, exist_ok=True)
    input_data = {
        'px': muons[:,0],
        'py': muons[:,1],
        'pz': muons[:,2],
        'x': muons[:,3],
        'y': muons[:,4],
        'z': muons[:,5]
    }

    for key, values in output.items():
        if key == 'pdg_id': continue
        plt.figure()
        # Plot input histogram as nonfilled
        plt.hist(input_data[key], bins=100, histtype='step', label='Input', linewidth=1.5, log=True)
        # Plot output histogram as nonfilled
        plt.hist(values, bins=100, histtype='step', label='Output', linewidth=1.5, log=True)
        plt.title(f"Histogram of {key} (CUDA_MUONS)")
        plt.xlabel(key)
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{key}_hist.png")
        plt.close()
    print(f"Histograms saved in {out_dir}")
