import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from MuonsAndMatter.cuda_muons.run_cuda_muons import propagate_muons_with_cuda, get_corners_from_detector, get_design_from_params, params_lib, get_cavern
import torch    
import h5py
import time


def run_cuda_simulation(muons, n_steps=5000, mag_field=[0., 0., 0.],histogram_dir = 'data', material = 'G4_Fe'):
    t1 = time.time()
    sensitive_film = {"name": "SensitiveFilm", "dz": 0.001, "dx": 20, "dy": 20,"shape": "plane"}
    muons = torch.from_numpy(muons)
    muons_momenta = muons[:,:3]
    muons_positions = muons[:,3:6]
    muons_charge = muons[:,6]

    if material == 'G4_Fe':
        hist_file = os.path.join(histogram_dir, 'alias_histograms_G4_Fe.pkl')
    elif material == 'G4_CONCRETE':
        hist_file = os.path.join(histogram_dir, 'alias_histograms_G4_CONCRETE.pkl')

    with open(hist_file, 'rb') as f:
        hist_data = pickle.load(f)
        histograms = [hist_data['hist_2d_probability_table'],hist_data['hist_2d_alias_table'],
                        hist_data['centers_first_dim'], hist_data['centers_second_dim'],
                        hist_data['width_first_dim'], hist_data['width_second_dim']]
        step_length = hist_data['step_length']
    histograms_sec = [torch.zeros_like(s) for s in histograms]

    detector = get_design_from_params(params = params_lib.params['only_HA'],
                      fSC_mag = False,
                      simulate_fields=False,
                      sensitive_film_params=None,
                      field_map_file = '/home/hep/lprate/projects/MuonsAndMatter/data/outputs/fields_mm.h5',
                      add_cavern = False,
                      add_target = True,
                      NI_from_B = True,
                      use_diluted = False,
                      SND = False)
    corners = get_corners_from_detector(detector)
    cavern = get_cavern(detector)

    if mag_field is None:
        mag_dict = detector['global_field_map']
        with h5py.File(os.path.join("/home/hep/lprate/projects/MuonsAndMatter", mag_dict['B']), 'r') as f:
            mag_dict['B'] = f["B"][:]
    else:
        mag_field = np.tile(np.asarray(mag_field), (116, 124, 620, 1)).reshape(-1,3).astype(np.float16)
        mag_dict = {'B': mag_field,'range_x':[0,230,2], 'range_y':[0,246,2], 'range_z':[-50,3045,5]}
    
    print("Starting propagation. Pre-definitions took:", time.time() - t1)
    t1 = time.time()
    out_position, out_momenta = propagate_muons_with_cuda (
            muons_positions,
            muons_momenta,
            muons_charge,
            corners,
            cavern,
            *histograms,
            *histograms_sec,
            mag_dict,
            n_steps*0.02 - sensitive_film['dz']/2,
            n_steps,
            step_length,)
    t2 = time.time()
    if n_steps > 1:
        in_sens_plane = (out_position[:,0].abs() < sensitive_film['dx']/2) & \
                        (out_position[:,1].abs() < sensitive_film['dy']/2) & \
                        (out_position[:,2] >= (n_steps*0.02 - sensitive_film['dz']/2))

        out_momenta = out_momenta[in_sens_plane]
        out_position = out_position[in_sens_plane]
        muons_charge = muons_charge[in_sens_plane].int()
    print("Number of outputs:", out_momenta.shape[0])


    out_position = out_position.cpu().numpy()
    out_momenta = out_momenta.cpu().numpy()
    output = {
        'px': out_momenta[:,0],
        'py': out_momenta[:,1],
        'pz': out_momenta[:,2],
        'x': out_position[:,0],
        'y': out_position[:,1],
        'z': out_position[:,2],
        'pdg_id': muons_charge*(-13)
    }
    print("Pos-propagation took:", time.time() - t2)
    return t2 - t1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='benchmark_times.pkl', help='Output file to save benchmark times')
    parser.add_argument('--mag_field', type=float, nargs=3, default=None, help='Magnetic field vector [Bx, By, Bz] in Tesla')
    args = parser.parse_args()
    

    with h5py.File("/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_after_target.h5", 'r') as f:
        px = f['px'][:]
        py = f['py'][:]
        pz = f['pz'][:]
        x = f['x'][:]
        y = f['y'][:]
        z = f['z'][:]
        pdg = f['pdg'][:]
        weight = f['weight'][:]
        muons = np.stack([px, py, pz, x, y, z, pdg, weight], axis=1).astype(np.float64)
    times = []
    n_muons_list = np.logspace(2, np.log10(500_000_000), 20)
    for n_muons in n_muons_list:
        n_muons = int(n_muons)
        print("=" * 60)
        print(f"Running for {n_muons} muons")
        dt = run_cuda_simulation(muons[:n_muons], n_steps = 5000, mag_field=args.mag_field, material='G4_Fe')
        times.append(dt)
    output_data = (n_muons_list, times)
    with open(os.path.join('data', args.output_file), 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Benchmark times saved to {args.output_file}")

